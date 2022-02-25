import torch
import numpy as np
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output

class Agent():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.device = torch.device(config['device'])

        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=config['betas'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config['lr_gamma'])

        self.pool = Pool()

    def step(self, x, target, steps, optimizer, scheduler, loss_function):
        x = self.model(x, steps=steps)
        loss = loss_function(x[:, :, :, 3:6], target[:, :, :, :3])
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

    def createSeed(self, target_img, config):
        pad_target = self.pad_target_f(target_img, config['target_padding'])
        h, w = pad_target.shape[1:3]
        seed = make_seed((h, w), config['channel_n'])# pad_target.cpu()#make_seed((h, w), CHANNEL_N)
        seed[:, :, 0:3] = pad_target.cpu()[0,:,:,:]
        return seed

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
                x0 = torch.from_numpy(batch_seed.astype(np.float32)).to(config['device'])
                batch_target = torch.from_numpy(batch_target.astype(np.float32)).to(config['device'])
                steps = self.getInferenceSteps(config)
                x, loss = self.step(x0, batch_target, steps, self.optimizer, self.scheduler, loss_function) #np.random.randint(64,96)
                loss_log.append(loss.item())
                if(config['Persistence'] == True):
                    pool_temp.addToPool(x.detach().cpu(), j, config, dataset)
            if(config['Persistence'] == True):
                self.pool = pool_temp
            if i%1 == 0:
                self.printIntermediateResults(config, loss, i)

class Pool():
    def __init__(self):
        self.pool = {}
        self.rng = np.random.default_rng(12345)
        return

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

    