#%%
import torch
#from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from src.models.Model_DiffusionNCA import DiffusionNCA
from src.models.Model_DiffusionNCA_Group import DiffusionNCA_Group
from src.models.Model_DiffusionNCA_fft import DiffusionNCA_fft
from src.models.Model_DiffusionNCA_fft2_sin import DiffusionNCA_fft2
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Diffusion import Agent_Diffusion
from src.datasets.Dataset_BCSS import Dataset_BCSS
from torch.profiler import profile, record_function, ProfilerActivity
import time

config = [{
    # Basic
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    #/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/
    'img_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/", #img_align_celeba, Emojis_Smiley, Emojis_Google
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/",
    'name': r'IGD_DiffusionNCA_Run19_CelebA_Normal_test', #last 58 #DiffusionNCA_Run585_CelebA_fixed_rescale_norm_fft_updat_l2_k7_multiNCA_4_smoothl1_twoStep
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4, #32
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 5,
    'evaluate_interval': 2,
    'n_epoch': 100000,
    'batch_size': 16,
    # Model
    'channel_n': 96,        # Number of CA state channels
    'batch_duplication': 1,
    'inference_steps': 20,
    'cell_fire_rate': 0.5,
    'input_channels': 3,
    'output_channels': 3,
    'hidden_size':  512,
    'schedule': 'linear',
    # Data
    'input_size': (64, 64),
    'data_split': [0.80340968, 0.09806, 1], 
    'timesteps': 300,
    '2D': True,
    'unlock_CPU': False,
}
]

#dataset = Dataset_NiiGz_3D(slice=2)
dataset = png_Dataset(buffer=True)
device = torch.device(config[0]['device'])
#ca = DiffusionNCA_Group(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)

ca1 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca2 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca3 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca4 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca5 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca6 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca7 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca8 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca9 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
ca0 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
ca = [ca0]#, ca1]#[ca0, ca1]#, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9]#, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9]

print("PARAMETERS", sum(p.numel() for p in ca0.parameters() if p.requires_grad))
 
agent = Agent_Diffusion(ca)
exp = Experiment(config, dataset, ca, agent)
#exp.bufferData()
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

def print_profiler_output(prof):
    for avg in prof.key_averages():
        print(
            f"{avg.key}: {avg.self_cpu_time_total} CPU time, "
            f"{avg.self_cuda_time_total} CUDA time, "
            f"{avg.cpu_memory_usage} CPU memory, "
            f"{avg.cuda_memory_usage} CUDA memory"
        )

if False:
    agent.train(data_loader, loss_function)
else:
    #torch.manual_seed(142)
    #agent.calculateFID_fromFiles(samples=100) #/home/jkalkhof_locale/Documents/GitHub/vnca2/Synth/
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
             record_shapes=True,
             profile_memory=True,  # Note: This can be very memory intensive
             with_stack=True) as prof:

        start_time = time.perf_counter()    
        #agent.test_fid(samples=1, optimized=False, saveImg=True)
        agent.generateSamples(samples=1, normal=True)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds to execute.")

    print_profiler_output(prof)
        


# %%
