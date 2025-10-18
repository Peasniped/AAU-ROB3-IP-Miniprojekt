import cv2
import torch

from components.board import Board

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    exit()

    board_index = 1
    board = Board(board_index)
    board.show(wait_key=False, destroy_windows=False)

    for tile in board.tiles:
        cv2.imshow(f"Tile {tile.index} (Col: {tile.col}, Row: {tile.row})", tile.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()