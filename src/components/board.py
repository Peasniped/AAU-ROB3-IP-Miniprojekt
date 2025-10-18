import cv2
import numpy as np
import json
from pathlib import Path

from components.tile import Tile

# Load settings
_settings_path = Path(__file__).parent.parent.parent / 'settings.json'
with open(_settings_path, 'r') as f:
    _settings = json.load(f)

class Board:

    def __init__(self, index:int):
        self.index: int = index
        base_path = _settings['board']['base_file_path'].replace('/', '\\')
        self.file_name: str = f"{base_path}\\{index}.jpg"
        self.image: np.ndarray = cv2.imread(self.file_name)
        self.image_hsv: np.ndarray = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.tiles = self._cut_into_tiles()

    def _cut_into_tiles(self) -> list[Tile]:
        board_width = _settings['board']['width']
        board_height = _settings['board']['height']
        tile_height = self.image.shape[0] // board_width
        tile_width = self.image.shape[1] // board_width
        margin = _settings['board']['tile_margin']

        tiles = []
        index = 0

        for row in range(board_height):
            for col in range(board_width):
                x1 = col * tile_width + margin
                x2 = (col + 1) * tile_width - margin
                y1 = row * tile_height + margin
                y2 = (row + 1) * tile_height - margin

                tile_image = self.image[y1:y2, x1:x2]

                tile = Tile(self, index, col, row, tile_image)
                tiles.append(tile)
                index += 1
        
        return tiles

    def get_tile_from_xy(self, col:int, row:int) -> Tile:
        for tile in self.tiles:
            if tile.col == col and tile.row == row:
                return tile
        return None
    
    def get_tile_from_index(self, index:int) -> Tile:
        if 0 <= index < len(self.tiles):
            return self.tiles[index]
        return None
    
    def get_image(self, color_space:str="BGR"):
        if color_space == "BGR":
            return self.image
        elif color_space == "HSV":
            return self.image_hsv
        else:
            raise ValueError("Unsupported color space. Use 'BGR' or 'HSV'.")
    
    def show(self, window_name:str="Board", color_space:str="BGR", wait_key:bool=True, destroy_windows:bool=True):
        cv2.imshow(window_name, self.get_image(color_space))
        if wait_key:
            cv2.waitKey(0)
        if destroy_windows:
            cv2.destroyAllWindows()
