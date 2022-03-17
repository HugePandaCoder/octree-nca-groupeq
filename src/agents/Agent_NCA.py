import torch
import numpy as np
from src.utils.helper import convert_image
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output


class Agent(BaseAgent):
    def __init__(self, model, exp):
        self.exp = exp
        self.model = model
        self.device = torch.device(self.exp.get_from_config('device'))
        self.optimizer = optim.Adam(model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
        self.pool = Pool()

    def loss_noOcillation(self, x, target, freeChange=True):
        #x = torch.flatten(x)
        if freeChange:
            x[x <= 1] = 0
            loss = x.sum() / torch.numel(x)
        else:
            xin_sum = torch.sum(x) + 1
            x = torch.square(target-x)
            loss = torch.sum(x) / xin_sum
        return loss

    r"""Execute a single step of the NCA
        Args:
            x (tensor): Input to model
            target (tensor): Target of model
            steps (int): Number of steps for inference
            optimizer (optim)
            scheduler (lr_scheduler)
            loss_function
    """
    def step(self, x, target, steps, optimizer, scheduler, loss_function):
        if self.exp.get_from_config('cell_fire_interval'):
            fire_rate_loc = np.random.uniform(low=self.exp.get_from_config('cell_fire_interval')[0], high=self.exp.get_from_config('cell_fire_interval')[1]) #np.random.Generator.uniform(low=self.config['cell_fire_interval'][0], high=self.config['cell_fire_interval'][1])
            x[...,3:] = self.model(x, steps=steps, fire_rate=fire_rate_loc)[...,3:]
        else:
            x_temp, x_sum = self.model(x, steps=steps)
            #x[...,3:] = x_temp[...,3:]
        loss = loss_function(x[:, :, :, 3:6], target[:, :, :, :3]) #+ self.loss_noOcillation(x_sum, None, freeChange=True) #torch.abs(target[:, :, :, :3] - x_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return x, loss

    r"""TODO 
    """
    def loss_f(self, x, target):
        return torch.mean(torch.pow(x[..., :3]-target, 2), [-2,-3,-1])

    r"""Creates a padding around the tensor 
        Args:
            target (tensor)
            padding (int): padding on all 4 sides
    """
    def pad_target_f(self, target, padding):
        target = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        return target

    r"""Create a seed for the NCA - TODO: Currently only 0 input
        Args:
            shape ([int, int]): height, width shape
            n_channels (int): Number of channels
    """
    def make_seed(self, shape, n_channels):
        seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
        #seed[shape[0]//2, shape[1]//2, 3:] = 1.0
        return seed

    r"""TODO: Describe 
    """
    def createSeed(self, target_img):
        pad_target = self.pad_target_f(target_img, self.exp.get_from_config('target_padding'))
        h, w = pad_target.shape[1:3]
        seed = self.make_seed((h, w), self.exp.get_from_config('channel_n'))# pad_target.cpu()#make_seed((h, w), CHANNEL_N)
        seed[:, :, 0:3] = pad_target.cpu()[0,:,:,:]
        return seed

    r"""Repeat batch -> Useful for better generalisation when doing random activated neurons
        Args:
            seed (tensor): Seed for NCA
            target (tensor): Target of Model
            repeat_factor (int): How many times it should be repeated
    """
    def repeatBatch(self, seed, target, repeat_factor):
        return np.repeat(seed, repeat_factor, axis=0), np.repeat(target, repeat_factor, axis=0)

    r"""Create a batch for use with the model
        Args:
            dataset (Dataset): To load the input from
            x (tensor): 
    """
    def makeBatch(self, dataset, x):
        # Create images
        batch_seed = np.empty([self.exp.get_from_config('batch_size'), self.exp.get_from_config('target_size') + 2* self.exp.get_from_config('target_padding'), self.exp.get_from_config('target_size') + 2* self.exp.get_from_config('target_padding'), self.exp.get_from_config('channel_n')])
        batch_target = np.empty([self.exp.get_from_config('batch_size'), self.exp.get_from_config('target_size') + 2* self.exp.get_from_config('target_padding'), self.exp.get_from_config('target_size') + 2* self.exp.get_from_config('target_padding'), 3])
        for j in range(self.exp.get_from_config('batch_size')):
            if(self.exp.get_from_config('Persistence') == True):
                target_img, target_label, in_pool = self.pool.getFromPool(j+x*self.exp.get_from_config('batch_size'), dataset)
                if in_pool == False:
                    seed = self.createSeed(target_img)
                else:
                    seed = target_img
            else:
                target_img, target_label = dataset.__getitem__((j+x*self.exp.get_from_config('batch_size')))
                seed = self.createSeed(target_img)
            pad_target_label = self.pad_target_f(target_label, self.exp.get_from_config('target_padding'))
            batch_seed[j] = seed
            batch_target[j] = pad_target_label.cpu()

        return batch_seed, batch_target

    r"""Get the number of steps for inference, if its set to an array its a random value inbetween
    """
    def getInferenceSteps(self):
        if len(self.exp.get_from_config('inference_steps')) == 2:
            steps = np.random.randint(self.exp.get_from_config('inference_steps')[0], self.exp.get_from_config('inference_steps')[1])
        else:
            steps = self.exp.get_from_config('inference_steps')[0]
        return steps
    
    r"""Prints intermediate results of training and adds it to tensorboard
        Args: 
            loss (torch)
            epoch (int) 
    """
    def printIntermediateResults(self, loss, epoch):
        clear_output()
        print(epoch, "loss =", loss.item())
        self.exp.save_model()
        self.exp.write_scalar('Loss/train', loss, epoch)

    r"""TODO: Use dataloader for more functionality instead, find way to do clean with Pool included (Second dataloader?)
    """
    def train(self, dataset, loss_function):
        loss_log = []
        for i in range(self.exp.currentStep, self.exp.get_max_steps()):
            if(self.exp.get_from_config('Persistence') == True):
                pool_temp = Pool()
            for j in range(int(np.floor(dataset.__len__()/self.exp.get_from_config('batch_size')))):
                batch_seed, batch_target = self.makeBatch(dataset, j)
                batch_seed, batch_target = self.repeatBatch(batch_seed, batch_target, self.exp.get_from_config('repeat_factor'))
                batch_seed = torch.from_numpy(batch_seed.astype(np.float32)).to(self.exp.get_from_config('device'))
                batch_target = torch.from_numpy(batch_target.astype(np.float32)).to(self.exp.get_from_config(('device')))
                steps = self.getInferenceSteps()
                output, loss = self.step(batch_seed, batch_target, steps, self.optimizer, self.scheduler, loss_function) #np.random.randint(64,96)
                if(self.exp.get_from_config('Persistence') == True):
                    pool_temp.addToPool(output.detach().cpu(), j*self.exp.get_from_config('batch_size'), self.exp, dataset)
            if(self.exp.get_from_config('Persistence') == True):
                self.pool = pool_temp
                print("Pool size: " + str(self.pool.__len__()))
            if i%1 == 0:
                self.printIntermediateResults(loss, self.exp.currentStep)
            if i%10 == 0:
                diceLoss = DiceLoss(useSigmoid=False)
                loss_dice = self.test(dataset, diceLoss, self.getInferenceSteps())
                self.exp.write_scalar('Dice/test', loss_dice, i)
            self.exp.increase_step()

    r"""Evaluate model on testdata by merging it into 3d volumes first
        Args:
            dataset (Dataset)
            loss_f (torch.nn.Module)
            steps (int): Number of steps to do for inference
    """
    def test(self, dataset, loss_f, steps=64):
        self.exp.set_model_state('test')
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0

        save_img = [5, 32, 45, 89, 357, 53, 122, 267, 97, 389]
        for i in range(dataset.__len__()):
            inputs, targets = dataset.__getitem__(i)
            inputs = self.createSeed(inputs)
            inputs, targets = torch.from_numpy(np.expand_dims(inputs, axis=0)), torch.from_numpy(np.expand_dims(targets, axis=0))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = torch.sigmoid(self.model.forward_x(inputs, steps=steps))

            _, id, slice = dataset.__getname__(i).split('_')
            #print(id)
            if id != patient_id and patient_id != None:
                loss = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0)
                print(patient_id + ", " + str(loss.item()))
                self.saveNiiGz(patient_3d_image, patient_3d_label, patient_id, self.exp.get_from_config('out_path'))
                patient_id, patient_3d_image, patient_3d_label = id, None, None
                average_loss, patient_count = average_loss + loss.item(), patient_count + 1
                print("Average Dice Loss: " + str(average_loss/patient_count))

            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs.detach().cpu()[:, :, :, 3] #:4
                patient_3d_label = targets.detach().cpu()[:, :, :, 0]
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()[:, :, :, 3])) #:4
                patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()[:, :, :, 0])) #:4
            # Add image to tensorboard
            if i in save_img: #np.random.random() < chance:
                self.exp.write_img('test/img/' + str(patient_id) + "_" + str(len(patient_3d_image)), convert_image(outputs[0].detach().cpu().numpy(), targets[0].detach().cpu().numpy(), encode_image=False), self.exp.currentStep+1)
        self.exp.set_model_state('train')
        return average_loss/patient_count

r"""Keeps the previous outputs of the model in stored in a pool
"""
class Pool():
    def __init__(self):
        self.pool = {}
        self.rng = np.random.default_rng() #12345
        return

    def __len__(self):
        return len(self.pool)

    r"""Add a value to the pool
        Args:
            output (tensor): Output to store
            idx (int): idx in dataset
            exp (Experiment): All experiment related configs
            dataset (Dataset): The dataset
    """
    def addToPool(self, output, idx, exp, dataset):
        for j in range(exp.get_from_config('batch_size')):
            if self.rng.random() < exp.get_from_config('pool_chance'):
                #print("Add to Pool")
                self.pool[dataset.getIdentifier(idx + j)] = output[j]

    r"""Get value from pool
        Args:
            item (int): idx of item
            dataset (Dataset)
    """
    def getFromPool(self, item, dataset):   
        target_img, target_label = dataset.__getitem__(item)
        id = dataset.getIdentifier(item)
        if id in self.pool:
            return self.pool[id], target_label, True
        else:
            return target_img, target_label, False

    