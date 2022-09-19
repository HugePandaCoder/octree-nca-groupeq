import os
import cv2

img_path = r"D:\Cityscape_Rain\leftImg8bit_rain\train"
out_path = r"D:\Cityscape_Rain\cityscape_smaller\images\train"
out_size = (400, 200)
isLabel = False

def recursiveFileSearch(path):
    for f in os.listdir(path):
        joined_path = os.path.join(path, f)
        if os.path.isdir(joined_path):
            recursiveFileSearch(joined_path)
        if os.path.isfile(joined_path):
            resizeImage(joined_path, f)

# https://stackoverflow.com/questions/8170982/strip-string-after-third-occurrence-of-character-python
def trunc_at(s, d, n=3):
    "Returns s truncated at the n'th (3rd by default) occurrence of the delimiter, d."
    return d.join(s.split(d, n)[:n])

def resizeImage(path, file_name):
    img = cv2.imread(path)
    #if "color" in path:
    if isLabel:
        img = cv2.resize(img, dsize=out_size, interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, dsize=out_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(out_path, trunc_at(file_name, "_") + ".png"), img)

if __name__ == '__main__':
    recursiveFileSearch(img_path)