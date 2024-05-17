


from matplotlib import pyplot as plt
from src.datasets import Dataset_Base
import torch
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.models.Model_OctreeNCA import OctreeNCA
from src.models.Model_OctreeNCA_3D import OctreeNCA3D
from src.utils.Experiment import Experiment
from torch.utils.data import DataLoader


@torch.no_grad()
def visualize(experiment: Experiment, dataset: Dataset_Base = None) -> plt.Figure:
    if isinstance(experiment.model, OctreeNCA):
        return visualize2d(experiment, dataset)
    else:
        return visualize3d(experiment, dataset)


@torch.no_grad()
def visualize2d(experiment: Experiment, dataset: Dataset_Base = None) -> plt.Figure:
    assert dataset is None, "not implemented yet"
    assert isinstance(experiment.agent, MedNCAAgent)
    assert isinstance(experiment.model, OctreeNCA)

    data = next(iter(experiment.data_loader))
    experiment.agent.prepare_data(data)
    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs)

    #convert to binary label
    gallery.append((gallery[-1] > 0.5).float())
    
    figure = plt.figure(figsize=(15, 5))
    #plot all figures in gallery
    for i, img in enumerate(gallery):
        plt.subplot(1, len(gallery)+1, i+1)
        plt.imshow(img[0, :, :, 1].cpu().numpy())
        plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
        plt.axis('off')
        
    plt.subplot(1, len(gallery)+1, i+2)
    plt.imshow(targets[0, 0, :, :].cpu().permute(1,0).numpy())
    plt.title(f"ground truth", fontsize=8)
    plt.axis('off')

    plt.savefig("inference_test.png", bbox_inches='tight')
    return figure


@torch.no_grad()
def visualize3d(experiment: Experiment, dataset: Dataset_Base = None) -> plt.Figure:
    assert isinstance(experiment.agent, M3DNCAAgent)
    assert isinstance(experiment.model, OctreeNCA3D)

    if dataset is None:
        data = next(iter(experiment.data_loader))
    else:
        loader = DataLoader(dataset)
        data = next(iter(loader))
    experiment.agent.prepare_data(data)
    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs) #list of BHWDC tensors

    #convert to binary label
    gallery.append((gallery[-1] > 0.5).float())
    
    figure = plt.figure(figsize=(15, 5))
    #plot all figures in gallery
    for i, img in enumerate(gallery):
        depth = img.shape[3]
        plt.subplot(1, len(gallery)+1, i+1)
        plt.imshow(img[0, :, :, depth//2, 1].cpu().numpy())
        plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
        plt.axis('off')
        
    plt.subplot(1, len(gallery)+1, i+2)
    #targets: BHWDC
    plt.imshow(targets[0, :, :, depth//2, 0].cpu().numpy())
    plt.title(f"ground truth", fontsize=8)
    plt.axis('off')
    return figure