import tables
import matplotlib.pyplot as plt

fname = "/local/scratch/clmn1/data/epi_processed/epistroma_train.pytable"

with tables.open_file(fname,'r') as db:
    print(db)
    print(db.root.img.shape)
    print(db.root.filename[0])
    plt.imshow(db.root.img[0])
    plt.imshow(db.root.mask[0], alpha=0.5)
    plt.show()

    for i in range(db.root.img.shape[0]):
        if not 1 in db.root.mask[i]:
            print("Image", i, "has no 1")