import os
import cv2

img_path = r"D:\Cityscape\leftImg8bit\val"
out_path = r"D:\Cityscape\cityscape_smaller\images\val"
out_size = (400, 200)
isLabel = False

def recursiveFileSearch(path):
    for f in os.listdir(path):
        joined_path = os.path.join(path, f)
        if os.path.isdir(joined_path):
            recursiveFileSearch(joined_path)
        if os.path.isfile(joined_path):
            resizeImage(joined_path, f)

def resizeImage(path, file_name):
    img = cv2.imread(path)
    #if "color" in path:
    if isLabel:
        img = cv2.resize(img, dsize=out_size, interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(out_path, file_name[:-16] + ".png"), img)

if __name__ == '__main__':
    recursiveFileSearch(img_path)