"""Run the final model on one complete board image.

Run from the repository root:
    python src\\predict_full.py --board 8
    python src\\predict_full.py --image boards\\full\\DSC_1263.JPG --points 470,315 1015,435 873,785 450,700
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from components.board import Board
from multitask_classifier import MultiTaskTileClassifier


ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / "settings.json").read_text())


def load_classes() -> list[str]:
    classes_path = ROOT / SETTINGS["paths"]["classes_file"]
    data = json.loads(classes_path.read_text())
    classes = data["classes"]
    if isinstance(classes, dict):
        return [classes[str(index)] for index in range(len(classes))]
    return classes


def parse_points(value: list[str]) -> np.ndarray:
    if len(value) != 4:
        raise ValueError("--points requires four corners: top-left top-right bottom-right bottom-left")
    points = []
    for point in value:
        coordinates = point.split(",")
        if len(coordinates) != 2:
            raise ValueError(f"Invalid point '{point}'. Use x,y.")
        points.append([float(coordinates[0]), float(coordinates[1])])
    return np.array(points, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict all tiles in one complete board image.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--board", type=int, help="Cropped board number, for example 8")
    source.add_argument("--image", type=Path, help="Full image containing a board")
    parser.add_argument(
        "--points",
        nargs=4,
        metavar=("TOP_LEFT", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"),
        help="Four board corners as x,y; required with --image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / SETTINGS["paths"]["model_dir"] / "tile_classifier_final.pth",
        help="Path to a model state-dict file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path; defaults to predictions/board_<number>.jpg",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.image is not None and args.points is None:
        parser.error("--points is required with --image because a full image may contain multiple boards")
    if args.board is not None and args.points is not None:
        parser.error("--points can only be used with --image")

    classes = load_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTileClassifier(
        num_tile_classes=len(classes),
        max_crowns=SETTINGS["model"]["max_crowns"],
    )
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_size = SETTINGS["model"].get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.board is not None:
        board = Board(args.board)
        board_image = board.image
        tiles = board.tiles
        output_stem = f"board_{args.board}"
    else:
        full_image = cv2.imread(str(args.image))
        if full_image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        points = parse_points(args.points)
        target_size = 1000
        destination = np.array(
            [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, destination)
        board_image = cv2.warpPerspective(full_image, matrix, (target_size, target_size))

        class BoardView:
            image = board_image

        board = BoardView()
        tiles = []
        tile_size = target_size // SETTINGS["board"]["width"]
        margin = SETTINGS["board"]["tile_margin"]
        for row in range(SETTINGS["board"]["height"]):
            for col in range(SETTINGS["board"]["width"]):
                x1, x2 = col * tile_size + margin, (col + 1) * tile_size - margin
                y1, y2 = row * tile_size + margin, (row + 1) * tile_size - margin

                class TileView:
                    pass

                tile = TileView()
                tile.index = row * SETTINGS["board"]["width"] + col
                tile.col, tile.row = col, row
                tile.image = board_image[y1:y2, x1:x2]
                tiles.append(tile)
        output_stem = Path(args.image).stem

    predictions = {}
    with torch.no_grad():
        for tile in tiles:
            image = Image.fromarray(tile.image[:, :, ::-1])
            image_tensor = transform(image).unsqueeze(0).to(device)
            class_output, crown_output = model(image_tensor)
            class_probabilities = torch.softmax(class_output, dim=1)
            crown_probabilities = torch.softmax(crown_output, dim=1)
            class_confidence, class_prediction = class_probabilities.max(1)
            crown_confidence, crown_prediction = crown_probabilities.max(1)
            predictions[tile.index] = {
                "class": classes[class_prediction.item()],
                "class_confidence": class_confidence.item(),
                "crowns": crown_prediction.item(),
                "crown_confidence": crown_confidence.item(),
            }

    output = board.image.copy()
    tile_width = output.shape[1] // SETTINGS["board"]["width"]
    tile_height = output.shape[0] // SETTINGS["board"]["height"]
    for tile in tiles:
        prediction = predictions[tile.index]
        confidence = min(prediction["class_confidence"], prediction["crown_confidence"])
        color = (0, 255, 0) if confidence >= 0.8 else (0, 255, 255) if confidence >= 0.5 else (0, 0, 255)
        x1, y1 = tile.col * tile_width, tile.row * tile_height
        x2, y2 = (tile.col + 1) * tile_width, (tile.row + 1) * tile_height
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"{prediction['class'][:3]} C{prediction['crowns']}"
        cv2.putText(output, text, (x1 + 5, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        print(
            f"Tile {tile.index:2d} ({tile.col}, {tile.row}): "
            f"{prediction['class']} | {prediction['crowns']} crown(s) | "
            f"class={prediction['class_confidence']:.1%}, crown={prediction['crown_confidence']:.1%}"
        )

    output_path = args.output or ROOT / "predictions" / f"{output_stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise IOError(f"Could not write prediction image: {output_path}")
    print(f"\nDevice: {device}")
    print(f"Annotated board saved to: {output_path}")


if __name__ == "__main__":
    main()
