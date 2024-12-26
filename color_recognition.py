import cv2
import numpy as np

# Function to get the name of the color
def get_color_name(R, G, B):
    colors = {
        (255, 0, 0): 'Red',
        (0, 255, 0): 'Green',
        (0, 0, 255): 'Blue',
        (255, 255, 0): 'Yellow',
        (255, 0, 255): 'Magenta',
        (0, 255, 255): 'Cyan',
        (255, 255, 255): 'White',
        (0, 0, 0): 'Black',
        (128, 128, 128): 'Gray',
        (128, 0, 0): 'Maroon',
        (128, 128, 0): 'Olive',
        (0, 128, 0): 'Dark Green',
        (128, 0, 128): 'Purple',
        (0, 128, 128): 'Teal',
        (0, 0, 128): 'Navy',
    }
    
    min_distance = float('inf')
    color_name = "Unknown"
    for color, name in colors.items():
        d = np.sqrt((color[0] - R) ** 2 + (color[1] - G) ** 2 + (color[2] - B) ** 2)
        if d < min_distance:
            min_distance = d
            color_name = name
    return color_name

