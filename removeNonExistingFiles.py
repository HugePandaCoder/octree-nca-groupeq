import os

img_path = "M:\MasterThesis\Datasets\VOC2012\JPEGImages_min"
label_path = "M:\MasterThesis\Datasets\VOC2012\SegmentationClass"

label_files = os.listdir(label_path)
label_files = [x[:-4] for x in label_files]
#print(label_files)

for f in os.listdir(img_path):
    if f[:-4] not in label_files:
        path = os.path.join(img_path, f)
        print("Remove: " + path)
        os.remove(path)
