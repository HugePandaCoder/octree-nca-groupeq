from src.utils.helper import saveNiiGz

import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image
from src.losses.LossFunctions import DiceLoss

class BaseAgent():
    def __init__(self, model):
        self.model = model

    def set_exp(self, exp):
        self.exp = exp
        self.initialize()

    def initialize(self):
        self.device = torch.device(self.exp.get_from_config('device'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

    r"""Prints intermediate results of training and adds it to tensorboard
        Args: 
            loss (torch)
            epoch (int) 
    """
    def printIntermediateResults(self, loss, epoch):
        #clear_output()
        print(epoch, "loss =", loss.item())
        self.exp.save_model()
        self.exp.write_scalar('Loss/train', loss, epoch)

    def prepare_data(self, data, eval=False):
        return data

    def get_outputs(self, data):
        return self.model(data)

    def initialize_epoch(self):
        return

    def conclude_epoch(self):
        return

    r"""Execute a single batch training step
        Args:
            data (tensor, tensor): inputs, targets
            loss_f (torch.nn.Module): loss function
        Returns:
            loss item
    """
    def batch_step(self, data, loss_f):
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = loss_f(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    r"""Write intermediate results to tensorboard
        Args:
            epoch (int): Current epoch
            los_log ([loss]): Array of losses
    """
    def intermediate_results(self, epoch, loss_log):
        average_loss = sum(loss_log) / len(loss_log)
        print(epoch, "loss =", average_loss)
        self.exp.write_scalar('Loss/train', average_loss, epoch)

    r"""Do an intermediate evluation during training 
        TODO: Make variable for more evaluation scores (Maybe pass list of metrics)
        Args:
            dataset (Dataset)
            epoch (int)
    """
    def intermediate_evaluation(self, dataloader, epoch):
        diceLoss = DiceLoss(useSigmoid=True)
        loss_log = self.test(diceLoss)
        self.exp.write_scalar('Dice/test', sum(loss_log.values())/len(loss_log), epoch)

    r"""Save state of current model
    """
    def save_state(self):
        self.exp.save_model()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    r"""Execute training of model
        Args:
            dataloader (Dataloader): contains training data
            loss_f (nn.Model): The loss for training"""
    def train(self, dataloader, loss_f):
        for epoch in range(self.exp.currentStep, self.exp.get_max_steps()):
            print("Epoch: " + str(epoch))
            loss_log = []
            self.initialize_epoch()
            print('Dataset size: ' + str(len(dataloader)))
            for i, data in enumerate(dataloader):
                loss_item = self.batch_step(data, loss_f)
                loss_log.append(loss_item)
            self.intermediate_results(epoch, loss_log)
            if epoch % self.exp.get_from_config('evaluate_interval') == 0:
                print("Evaluate model")
                self.intermediate_evaluation(dataloader, epoch)
            if epoch % self.exp.get_from_config('save_interval') == 0:
                print("Model saved")
                self.save_state()
            self.conclude_epoch()
            self.exp.increase_epoch()

    def prepare_image_for_display(self, image):
        return image

    r"""Evaluate model on testdata by merging it into 3d volumes first
        TODO: Write nicely with dataloader
        Args:
            dataset (Dataset)
            loss_f (torch.nn.Module)
            steps (int): Number of steps to do for inference
    """
    def test(self, loss_f):
        dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        loss_log = {}

        save_img = [5, 32, 45, 89, 357, 53, 122, 267, 97, 389]
        for i, data in enumerate(dataloader):
            #data = dataset.__getitem__(i)
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data
            outputs, targets = self.get_outputs(data)

            if isinstance(data_id, str):
                print("IsInstance")
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                _, id, slice = data_id[0].split('_')

            #print(id)
            if id != patient_id and patient_id != None:
                loss_log[id] = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0).item()
                print(patient_id + ", " + str(loss_log[id]))
                patient_id, patient_3d_image, patient_3d_label = id, None, None

            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs.detach().cpu()#[:, :, :, 0] #:4
                patient_3d_label = targets.detach().cpu()#[:, :, :, 0]
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))#[:, :, :, 0])) #:4
                patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))#[:, :, :, 0])) #:4
            # Add image to tensorboard
            if i in save_img: #np.random.random() < chance:
                self.exp.write_img('test/img/' + str(patient_id) + "_" + str(len(patient_3d_image)), 
                convert_image(self.prepare_image_for_display(inputs.detach().cpu()).numpy(), 
                self.prepare_image_for_display(outputs.detach().cpu()).numpy(), 
                self.prepare_image_for_display(targets.detach().cpu()).numpy(), 
                encode_image=False), self.exp.currentStep)

        loss_log[id] = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0).item()
        print("Average Dice Loss: " + str(sum(loss_log.values())/len(loss_log)))

        self.exp.set_model_state('train')
        return loss_log