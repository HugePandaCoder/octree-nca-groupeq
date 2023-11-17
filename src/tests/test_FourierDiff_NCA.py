import torch
from ..datasets.png_Dataset import png_Dataset
from ..models.Model_DiffusionNCA_fft2_sin import DiffusionNCA_fft2
from ..losses.LossFunctions import DiceBCELoss
from ..utils.Experiment import Experiment
from ..agents.Agent_Diffusion import Agent_Diffusion
from ..utils.ProjectConfiguration import ProjectConfiguration
import tempfile
import os
import numpy as np
from PIL import Image

def create_testdata():
    image_shape = (32, 32)

    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories for noise data and ones and zeros
    start_dir = os.path.join(temp_dir, "start_dir")
    target_dir = os.path.join(temp_dir, "target_dir")

    os.makedirs(start_dir)
    os.makedirs(target_dir)

    # Create temporary png files for start
    for i in range(5):
        filename = os.path.join(start_dir, f"test_{i}.png")
        start_data = np.zeros(shape=image_shape, dtype=np.uint8)
        start_data[start_data.shape[0]//2, start_data.shape[1]//2] = 1
        start_image = Image.fromarray(start_data)#, mode='RGB')
        start_image.save(filename)

    # Create temporary png files for target
    for i in range(5):
        filename = os.path.join(target_dir, f"test_{i}.png")
        start_data = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)
        start_image = Image.fromarray(start_data)#, mode='RGB')
        start_image.save(filename)

    return start_dir, target_dir

def test_FourierDiffNCA():
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()

    config = [{
        # Basic
        'img_path': path_img,
        'label_path': path_label, 
        'name': r'test_diffusion', 
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr': 16e-4, 
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 1,
        'evaluate_interval': 1,
        'n_epoch': 1,
        'batch_size': 16,
        # Model
        'channel_n': 32,        # Number of CA state channels
        'batch_duplication': 1,
        'inference_steps': 10,
        'cell_fire_rate': 0.5,
        'input_channels': 3,
        'output_channels': 3,
        'hidden_size':  128,
        'schedule': 'linear',
        # Data
        'input_size': (32, 32),
        'data_split': [0.7, 0, 0.3], 
        'timesteps': 13,
        '2D': True,
        'unlock_CPU': True,
    }
    ]

    dataset = png_Dataset(buffer=True)
    device = torch.device(config[0]['device'])

    ca0 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
    ca = [ca0]
    
    agent = Agent_Diffusion(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceBCELoss() 

    agent.train(data_loader, loss_function)
    agent.test_fid(samples=1, optimized=False, saveImg=True)
    #agent.test_fid(samples=1, optimized=True, saveImg=True)
    agent.generateSamples(samples=1, normal=True)