import torch
import numpy as np
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from src.agents.Agent import BaseAgent

class Agent(BaseAgent):

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.device = torch.device(config['device'])

        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=config['betas'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config['lr_gamma'])

    def train(self, dataloader, epochs, loss_f):
        for epoch in range(epochs):
            print("Epoch: " + str(epoch))
            loss_log = []
            for i, data in enumerate(dataloader):
                inputs, targets = data
                inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = targets.permute(0, 3, 1, 2)
                outputs = self.model(inputs)

                self.optimizer.zero_grad()
                loss = loss_f(outputs[:,0,:,:], targets[:,0,:,:])
                loss_log.append(loss.item())
                loss.backward()
                self.optimizer.step()

            print(epoch, "loss =", loss.item())

            if epoch % 10 == 0:
                print("Model saved")
                torch.save(self.model.state_dict(), self.config['model_path'])

        return

    def test(self, dataset, config, loss_f, steps=64):
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0

        for i in range(dataset.__len__()):
            inputs, targets = dataset.__getitem__(i)
            #inputs = self.createSeed(inputs, config)
            inputs, targets = torch.from_numpy(np.expand_dims(inputs, axis=0)), torch.from_numpy(np.expand_dims(targets, axis=0))
            inputs, targets = inputs.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2)
            outputs = self.model(inputs) 

            _, id, slice = dataset.__getname__(i).split('_')
            #print(id)
            if id != patient_id and patient_id != None:
                loss = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0)
                self.saveNiiGz(patient_3d_image, patient_3d_label, patient_id, config['out_path'])
                print(patient_id + ", " + str(loss.item()))
                patient_id, patient_3d_image, patient_3d_label = id, None, None
                average_loss, patient_count = average_loss + loss.item(), patient_count + 1
            
            #print(outputs)
            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs[:, 0, :, :] #.permute(0, 2, 3, 1) #:4
                patient_3d_label = targets[:, 0, :, :] #.permute(0, 2, 3, 1)
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs[:, 0, :, :])) #:4
                patient_3d_label = torch.vstack((patient_3d_label, targets[:, 0, :, :])) #:4
        print("Average Loss: " + str(average_loss/patient_count))
        return