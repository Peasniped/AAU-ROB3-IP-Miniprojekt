import cv2
import torch
import json

from train.auto_labeller import AutoLabeller

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda._get_device(0)}")
    
    # First, validate the labels file
    print("\n=== Validating Labels ===")
    with open("src/board_labels.json", "r") as f:
        labels = json.load(f)
    
    max_tile_class = -1
    max_crown_count = -1
    invalid_boards = []
    
    for board_idx, board_labels in labels.items():
        for tile_idx, tile_data in board_labels.items():
            tile_class = tile_data["tile_class"]
            crown_count = tile_data["crown_count"]
            
            if tile_class > max_tile_class:
                max_tile_class = tile_class
            if crown_count > max_crown_count:
                max_crown_count = crown_count
            
            # Check for invalid values (0-8 for classes, 0-3 for crowns)
            if tile_class < 0 or tile_class > 8:
                invalid_boards.append((board_idx, tile_idx, "tile_class", tile_class))
            if crown_count < 0 or crown_count > 3:
                invalid_boards.append((board_idx, tile_idx, "crown_count", crown_count))
    
    print(f"Max tile_class found: {max_tile_class} (expected 0-8)")
    print(f"Max crown_count found: {max_crown_count} (expected 0-3)")
    
    if invalid_boards:
        print("\n⚠️ INVALID LABELS FOUND:")
        for board_idx, tile_idx, field, value in invalid_boards:
            print(f"  Board {board_idx}, Tile {tile_idx}: {field} = {value}")
        print("\nPlease fix these values in board_labels.json before training.")
        exit(1)
    
    print("✓ All labels valid!")
    
    # Start auto-labelling workflow
    labeller = AutoLabeller(labels_file="src/board_labels.json")
    
    # Run auto-labelling - with Human In The Loop (HITL)
    #labeller.run_auto_labelling(total_boards=40)
    labeller.run_auto_labelling(total_boards=40, training_boards_num=1)