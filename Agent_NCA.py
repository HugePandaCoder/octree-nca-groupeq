import torch
import numpy as np
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from Agent import BaseAgent

class Agent(BaseAgent):
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.device = torch.device(config['device'])

        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=config['betas'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config['lr_gamma'])

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

    def step(self, x, target, steps, optimizer, scheduler, loss_function):
        #x_mask = x[:, :, :, 3:6]
        if self.config['cell_fire_interval']:
            fire_rate_loc = np.random.uniform(low=self.config['cell_fire_interval'][0], high=self.config['cell_fire_interval'][1]) #np.random.Generator.uniform(low=self.config['cell_fire_interval'][0], high=self.config['cell_fire_interval'][1])
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

    def loss_f(self, x, target):
        return torch.mean(torch.pow(x[..., :3]-target, 2), [-2,-3,-1])

    def pad_target_f(self, target, padding):
        target = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        return target

    def make_seed(self, shape, n_channels):
        seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
        #seed[shape[0]//2, shape[1]//2, 3:] = 1.0
        return seed

    def createSeed(self, target_img, config):
        pad_target = self.pad_target_f(target_img, config['target_padding'])
        h, w = pad_target.shape[1:3]
        seed = self.make_seed((h, w), config['channel_n'])# pad_target.cpu()#make_seed((h, w), CHANNEL_N)
        seed[:, :, 0:3] = pad_target.cpu()[0,:,:,:]
        return seed

    def repeatBatch(self, seed, target, repeat_factor):
        return np.repeat(seed, repeat_factor, axis=0), np.repeat(target, repeat_factor, axis=0)

    def makeBatch(self, dataset, config, x):
        # Create images
        batch_seed = np.empty([config['batch_size'], config['target_size'] + 2* config['target_padding'], config['target_size'] + 2* config['target_padding'], config['channel_n']])
        batch_target = np.empty([config['batch_size'], config['target_size'] + 2* config['target_padding'], config['target_size'] + 2* config['target_padding'], 3])
        for j in range(config['batch_size']):
            if(config['Persistence'] == True):
                target_img, target_label, in_pool = self.pool.getFromPool(config, j+x*config['batch_size'], dataset)
                if in_pool == False:
                    seed = self.createSeed(target_img, config)
                else:
                    seed = target_img
            else:
                target_img, target_label = dataset.__getitem__((j+x*config['batch_size']))
                seed = self.createSeed(target_img, config)
            pad_target_label = self.pad_target_f(target_label, config['target_padding'])
            batch_seed[j] = seed
            batch_target[j] = pad_target_label.cpu()

        return batch_seed, batch_target

    def getInferenceSteps(self, config):
        if len(config['inference_steps']) == 2:
            steps = np.random.randint(config['inference_steps'][0],config['inference_steps'][1])
        else:
            steps = config['inference_steps'][0]
        return steps
    
    def printIntermediateResults(self, config, loss, i):
        clear_output()
        print(i, "loss =", loss.item())
        torch.save(self.model.state_dict(), config['model_path'])

    def train(self, dataset, config, loss_function):
        loss_log = []
        for i in range(config['n_epoch']):
            if(config['Persistence'] == True):
                pool_temp = Pool()
            for j in range(int(np.floor(dataset.__len__()/config['batch_size']))):
                batch_seed, batch_target = self.makeBatch(dataset, config, j)
                batch_seed, batch_target = self.repeatBatch(batch_seed, batch_target, config['repeat_factor'])
                batch_seed = torch.from_numpy(batch_seed.astype(np.float32)).to(config['device'])
                batch_target = torch.from_numpy(batch_target.astype(np.float32)).to(config['device'])
                steps = self.getInferenceSteps(config)
                output, loss = self.step(batch_seed, batch_target, steps, self.optimizer, self.scheduler, loss_function) #np.random.randint(64,96)
                #loss_log.append(loss.item())
                if(config['Persistence'] == True):
                    pool_temp.addToPool(output.detach().cpu(), j*config['batch_size'], config, dataset)
            if(config['Persistence'] == True):
                self.pool = pool_temp
                print("Pool size: " + str(self.pool.__len__()))
            if i%1 == 0:
                self.printIntermediateResults(config, loss, i)

    def test(self, dataset, config, loss_f, steps=64):
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0

        for i in range(dataset.__len__()):
            inputs, targets = dataset.__getitem__(i)
            inputs = self.createSeed(inputs, config)
            inputs, targets = torch.from_numpy(np.expand_dims(inputs, axis=0)), torch.from_numpy(np.expand_dims(targets, axis=0))
            outputs = torch.sigmoid(self.model.forward_x(inputs, steps=steps))

            _, id, slice = dataset.__getname__(i).split('_')
            #print(id)
            if id != patient_id and patient_id != None:
                loss = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0)
                print(patient_id + ", " + str(loss.item()))
                self.saveNiiGz(patient_3d_image, patient_3d_label, patient_id, config['out_path'])
                patient_id, patient_3d_image, patient_3d_label = id, None, None
                average_loss, patient_count = average_loss + loss.item(), patient_count + 1
                print("Average Loss: " + str(average_loss/patient_count))

            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs[:, :, :, 3] #:4
                patient_3d_label = targets[:, :, :, 0]
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs[:, :, :, 3])) #:4
                patient_3d_label = torch.vstack((patient_3d_label, targets[:, :, :, 0])) #:4
        return

class Pool():
    def __init__(self):
        self.pool = {}
        self.rng = np.random.default_rng() #12345
        return

    def __len__(self):
        return len(self.pool)

    def addToPool(self, output, idx, config, dataset):
        for j in range(config['batch_size']):
            if self.rng.random() < config['pool_chance']:
                #print("Add to Pool")
                self.pool[dataset.getIdentifier(idx + j)] = output[j]

    def getFromPool(self, config, item, dataset):   
        target_img, target_label = dataset.__getitem__(item)
        id = dataset.getIdentifier(item)
        if id in self.pool:
            return self.pool[id], target_label, True
        else:
            return target_img, target_label, False

    