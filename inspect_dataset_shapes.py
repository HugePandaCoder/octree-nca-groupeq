
import os
import nibabel as nib
import numpy as np

def compute_num_levels_and_input_dim(dim):
    num_levels = np.ceil(np.log2(dim / 16))
    dim_lowest_resolution = 0
    input_dim = np.ceil(dim / 2**num_levels) * 2**num_levels
    return int(num_levels), int(input_dim)


def print_hippocampus_info():
    hippocampus = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/imagesTr/"

    shapes = []
    for file in os.listdir(hippocampus):
        x = nib.load(os.path.join(hippocampus, file)).get_fdata()
        
        shapes.append(x.shape)

    shapes = np.array(shapes)
    min_shape = shapes.min(axis=0)
    max_shape = shapes.max(axis=0)
    print(f"Min shape: {min_shape}")
    print(f"Max shape: {max_shape}")

    num_levels_and_input_dims = [compute_num_levels_and_input_dim(dim) for dim in max_shape]
    num_levels_and_input_dims = np.array(num_levels_and_input_dims)
    
    print(f"Ideal input shape: {num_levels_and_input_dims[:,1]}")


def print_prostate_info():
    hippocampus = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/imagesTr/"

    shapes = []
    for file in os.listdir(hippocampus):
        x = nib.load(os.path.join(hippocampus, file)).get_fdata()
        
        shapes.append(x.shape)

    shapes = np.array(shapes)
    min_shape = shapes.min(axis=0)
    max_shape = shapes.max(axis=0)
    print(f"Min shape: {min_shape}")
    print(f"Max shape: {max_shape}")

    num_levels_and_input_dims = [compute_num_levels_and_input_dim(dim) for dim in max_shape]
    num_levels_and_input_dims = np.array(num_levels_and_input_dims)
    
    print(f"Ideal input shape: {num_levels_and_input_dims[:,1]}")

if __name__ == "__main__":
    print_prostate_info()