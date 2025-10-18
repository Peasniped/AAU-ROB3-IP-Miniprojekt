import cv2

class Tile:
    def __init__(self, parent_board, index:int, col:int, row:int, image):
        self.parent_board = parent_board
        self.index = index
        self.col = col
        self.row = row
        self.image = image
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)