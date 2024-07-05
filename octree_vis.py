


from matplotlib import pyplot as plt
from src.datasets import Dataset_Base
import torch
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.models.Model_OctreeNCA import OctreeNCA
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
from src.models.Model_OctreeNCA_3D import OctreeNCA3D
from src.utils.Experiment import Experiment
import torch.utils.data


def find_sample_by_id(experiment, dataset: Dataset_Base, id:str):
    assert False, "not implemented yet"
    for split in ['train', 'val', 'test']:
        if len(experiment.data_split.get_images(split)) == 0:
            continue
        experiment.dataset.setPaths(experiment.config['img_path'], experiment.data_split.get_images(split), 
                                    experiment.config['label_path'], experiment.data_split.get_labels(split))
        experiment.dataset.setState(split)

        print(experiment.dataset.images_list)

        try:
            item = dataset.getItemByName((id, id, 0))
            return item
        except ValueError:
            pass
    assert False, f"Could not find sample with id {id}"

@torch.no_grad()
def visualize(experiment: Experiment, dataset: Dataset_Base = None, split: str='test', sample_id: str=None) -> plt.Figure:
    
    if dataset is None:
        if sample_id is None:
            loader = torch.utils.data.DataLoader(experiment.datasets[split])
            data = next(iter(loader))
        else:
            data = find_sample_by_id(experiment, experiment.datasets[split], sample_id)
    else:
        if sample_id is None:
            loader = torch.utils.data.DataLoader(dataset)
            data = next(iter(loader))
        else:
            data = find_sample_by_id(experiment, dataset, sample_id)


    if isinstance(experiment.model, OctreeNCA2DPatch2):
        return visualize2d(experiment, data)
    else:
        return visualize3d(experiment, data)


@torch.no_grad()
def visualize2d(experiment: Experiment, data: dict) -> plt.Figure:
    assert isinstance(experiment.agent, MedNCAAgent)
    assert isinstance(experiment.model, OctreeNCA2DPatch2)

    data = experiment.agent.prepare_data(data)

    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs)

    #convert to binary label
    gallery.append((gallery[-1] > 0.5).float())
    
    figure = plt.figure(figsize=(15, 5))
    #plot all figures in gallery
    for i, img in enumerate(gallery):
        plt.subplot(1, len(gallery)+1, i+1)
        plt.imshow(img[0, :, :, 3].cpu().numpy())
        plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
        plt.axis('off')
        
    plt.subplot(1, len(gallery)+1, i+2)
    plt.imshow(targets[0, 0, :, :].cpu().permute(1,0).numpy())
    plt.title(f"ground truth", fontsize=8)
    plt.axis('off')

    plt.savefig("inference_test.png", bbox_inches='tight')
    return figure


@torch.no_grad()
def visualize3d(experiment: Experiment, data) -> plt.Figure:
    assert isinstance(experiment.agent, M3DNCAAgent)
    assert isinstance(experiment.model, OctreeNCA3D)

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