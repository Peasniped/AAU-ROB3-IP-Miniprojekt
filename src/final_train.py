"""Train and evaluate the multi-task tile model with board-level splits.

Run from the repository root:
    python src\final_train.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from components.board import Board
from multitask_classifier import MultiTaskTileClassifier


ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / "settings.json").read_text())


class TileDataset(Dataset):
    def __init__(self, tiles: list[dict], transform: transforms.Compose) -> None:
        self.tiles = tiles
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        item = self.tiles[index]
        image = Image.fromarray(item["image"][:, :, ::-1])
        return (
            self.transform(image),
            int(item["tile_class"]),
            int(item["crown_count"]),
        )


def load_labels() -> dict:
    path = ROOT / SETTINGS["paths"]["labels_file"]
    return json.loads(path.read_text())


def split_boards(board_ids: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    if len(board_ids) < 3:
        raise ValueError("At least three labelled boards are required for train/validation/test splits.")

    rng = random.Random(seed)
    shuffled = board_ids[:]
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.15))
    validation_count = max(1, round(len(shuffled) * 0.15))
    test = shuffled[:test_count]
    validation = shuffled[test_count:test_count + validation_count]
    train = shuffled[test_count + validation_count:]
    return sorted(train), sorted(validation), sorted(test)


def make_tiles(labels: dict, board_ids: list[int]) -> list[dict]:
    tiles = []
    for board_id in board_ids:
        board = Board(board_id)
        for tile_id, label in labels[str(board_id)].items():
            tile = board.get_tile_from_index(int(tile_id))
            if tile is None:
                raise ValueError(f"Board {board_id} has no tile {tile_id}.")
            tiles.append({
                "image": tile.image,
                "tile_class": label["tile_class"],
                "crown_count": label["crown_count"],
            })
    return tiles


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    class_correct = crown_correct = total = total_loss = 0
    class_loss = nn.CrossEntropyLoss()
    crown_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, classes, crowns in loader:
            images, classes, crowns = images.to(device), classes.to(device), crowns.to(device)
            class_output, crown_output = model(images)
            total_loss += (class_loss(class_output, classes) + crown_loss(crown_output, crowns)).item()
            class_correct += (class_output.argmax(1) == classes).sum().item()
            crown_correct += (crown_output.argmax(1) == crowns).sum().item()
            total += images.size(0)
    batches = max(1, len(loader))
    return total_loss / batches, class_correct / total, crown_correct / total


def train(
    model: nn.Module,
    loader: DataLoader,
    validation_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["training"]["learning_rate"])
    class_loss = nn.CrossEntropyLoss()
    crown_loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, classes, crowns in loader:
            images, classes, crowns = images.to(device), classes.to(device), crowns.to(device)
            optimizer.zero_grad()
            class_output, crown_output = model(images)
            loss = class_loss(class_output, classes) + crown_loss(crown_output, crowns)
            loss.backward()
            optimizer.step()
        if validation_loader is not None:
            loss_value, class_accuracy, crown_accuracy = evaluate(model, validation_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs} - validation loss: {loss_value:.4f} - "
                f"class accuracy: {class_accuracy:.2%} - crown accuracy: {crown_accuracy:.2%}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the tile model.")
    parser.add_argument("--epochs", type=int, default=SETTINGS["training"]["initial_epochs"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = load_labels()
    board_ids = sorted(int(board_id) for board_id in labels)
    train_ids, validation_ids, test_ids = split_boards(board_ids, args.seed)
    print(f"Train boards: {train_ids}")
    print(f"Validation boards: {validation_ids}")
    print(f"Test boards: {test_ids}")

    image_size = SETTINGS["model"].get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    batch_size = SETTINGS["training"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TileDataset(make_tiles(labels, train_ids), transform), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        TileDataset(make_tiles(labels, validation_ids), transform), batch_size=batch_size
    )
    test_loader = DataLoader(TileDataset(make_tiles(labels, test_ids), transform), batch_size=batch_size)

    model = MultiTaskTileClassifier(len(json.loads((ROOT / SETTINGS["paths"]["classes_file"]).read_text())["classes"]), SETTINGS["model"]["max_crowns"]).to(device)
    print(f"\nTraining on train boards using {device}...")
    train(model, train_loader, validation_loader, device, args.epochs)
    loss, class_accuracy, crown_accuracy = evaluate(model, test_loader, device)
    print(f"Test results - loss: {loss:.4f}, class accuracy: {class_accuracy:.2%}, crown accuracy: {crown_accuracy:.2%}")

    final_loader = DataLoader(
        TileDataset(make_tiles(labels, train_ids + validation_ids), transform),
        batch_size=batch_size,
        shuffle=True,
    )
    final_model = MultiTaskTileClassifier(
        len(json.loads((ROOT / SETTINGS["paths"]["classes_file"]).read_text())["classes"]),
        SETTINGS["model"]["max_crowns"],
    ).to(device)
    print("\nTraining final model on train + validation boards...")
    train(final_model, final_loader, None, device, args.epochs)
    loss, class_accuracy, crown_accuracy = evaluate(final_model, test_loader, device)
    print(
        f"Final held-out test results - loss: {loss:.4f}, "
        f"class accuracy: {class_accuracy:.2%}, crown accuracy: {crown_accuracy:.2%}"
    )

    output_path = ROOT / SETTINGS["paths"]["model_dir"] / "tile_classifier_final.pth"
    torch.save(final_model.state_dict(), output_path)
    print(f"Final model saved to {output_path}")


if __name__ == "__main__":
    main()
