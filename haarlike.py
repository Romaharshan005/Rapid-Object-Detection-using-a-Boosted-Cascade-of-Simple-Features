import numpy as np
import cv2
import time
from integralImage import integralImage as ii

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)
    
def get_all_haarlike_features(height, width):
    features = []
    for w in range(1, height+1):
        for h in range(1, width+1):
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    #2 rectangle features
                    immediate = RectangleRegion(i, j, w, h)
                    right = RectangleRegion(i+w, j, w, h)
                    if i + 2 * w < width: #Horizontally Adjacent
                        features.append(([right], [immediate]))

                    bottom = RectangleRegion(i, j+h, w, h)
                    if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                    right_2 = RectangleRegion(i+2*w, j, w, h)
                    #3 rectangle features
                    if i + 3 * w < width: #Horizontally Adjacent
                        features.append(([right], [right_2, immediate]))

                    bottom_2 = RectangleRegion(i, j+2*h, w, h)
                    if j + 3 * h < height: #Vertically Adjacent
                        features.append(([bottom], [bottom_2, immediate]))

                    #4 rectangle features
                    bottom_right = RectangleRegion(i+w, j+h, w, h)
                    if i + 2 * w < width and j + 2 * h < height:
                        features.append(([right, bottom], [immediate, bottom_right]))

                    j += 1
                i += 1
    return np.array(features)