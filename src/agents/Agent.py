from src.utils.helper import saveNiiGz

import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image
from src.losses.LossFunctions import DiceLoss
import seaborn as sns

class BaseAgent():
    """Base class for all agents. Handles basic training and only needs to be adapted if special use cases are necessary.
    
    .. note:: In many cases only the data preparation and outputs need to be changed."""
    def __init__(self, model):
        self.model = model

    def set_exp(self, exp):
        r"""Set experiment of agent and initialize.
            Args:
                exp (Experiment): Experiment class"""
        self.exp = exp
        self.initialize()

    def initialize(self):
        self.device = torch.device(self.exp.get_from_config('device'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

    def printIntermediateResults(self, loss, epoch):
        r"""Prints intermediate results of training and adds it to tensorboard
            Args: 
                loss (torch)
                epoch (int) 
        """
        #clear_output()
        print(epoch, "loss =", loss.item())
        self.exp.save_model()
        self.exp.write_scalar('Loss/train', loss, epoch)

    def prepare_data(self, data, eval=False):
        r"""If any data preparation needs to be done do it here. 
            Args:
                data ([]): The data to be processed.
                eval (Bool): Whether or not its for evaluation. 
        """
        return data

    def get_outputs(self, data):
        r"""Get the output of the model.
            Args: 
                data (torch): The data to be passed to the model.
        """
        return self.model(data)

    def initialize_epoch(self):
        r"""Everything that should happen once before each epoch should be defined here.
        """
        return

    def conclude_epoch(self):
        r"""Everything that should happen once after each epoch should be defined here.
        """
        return

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            Args:
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            Returns:
                loss item
        """
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        #targets = targets.int()
        loss = loss_f(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def intermediate_results(self, epoch, loss_log):
        r"""Write intermediate results to tensorboard
            Args:
                epoch (int): Current epoch
                los_log ([loss]): Array of losses
        """
        average_loss = sum(loss_log) / len(loss_log)
        print(epoch, "loss =", average_loss)
        self.exp.write_scalar('Loss/train', average_loss, epoch)

    def plot_results_byPatient(self, loss_log):
        r"""Plot losses in a per patient fashion with seaborn to display in tensorboard.
            Args:
                loss_log ({name: loss}: Dictionary of losses
        """
        #loss_log = np.array(loss_log)
        print(loss_log)
        sns.set_theme()
        plot = sns.scatterplot(x=loss_log.keys(), y=loss_log.values())
        plot.set(ylim=(0, 1))
        plot = plot.get_figure()
        return plot

    def intermediate_evaluation(self, dataloader, epoch):
        r"""Do an intermediate evluation during training 
            .. todo:: Make variable for more evaluation scores (Maybe pass list of metrics)
            Args:
                dataset (Dataset)
                epoch (int)
        """
        diceLoss = DiceLoss(useSigmoid=True)
        loss_log = self.test(diceLoss)
        img_plot = self.plot_results_byPatient(loss_log)
        self.exp.write_figure('Patient/dice', img_plot, epoch)
        self.exp.write_scalar('Dice/test', sum(loss_log.values())/len(loss_log), epoch)
        self.exp.write_histogram('Dice/test/byPatient', np.fromiter(loss_log.values(), dtype=float), epoch)
        param_lst = []
        for param in self.model.parameters():
            #print(param.flatten())
            param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #print(param_lst)
        self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)

    def getAverageDiceScore(self, useSigmoid=True):
        r"""Get the average Dice test score.
            Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=[])
        return sum(loss_log.values())/len(loss_log)

    def save_state(self, model_path):
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))

    def load_state(self, model_path):
        r"""Load state of current model
        """
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth')))
        self.scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pth')))

    def train(self, dataloader, loss_f):
        r"""Execute training of model
            Args:
                dataloader (Dataloader): contains training data
                loss_f (nn.Model): The loss for training"""
        for epoch in range(self.exp.currentStep, self.exp.get_max_steps()+1):
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
                self.save_state(os.path.join(self.exp.get_from_config('model_path'), 'models', 'epoch_' + str(self.exp.currentStep)))
            self.conclude_epoch()
            self.exp.increase_epoch()

    def prepare_image_for_display(self, image):
        r"""Prepare an image to be displayed in tensorboard. Since images need to be in a specific format these modifications these can be done here.
            Args:
                image (torch): The image to be processed for display. 
        """
        return image

    def test(self, loss_f, save_img = None):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            Args:
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        loss_log = {}

        if save_img == None:
            save_img = [5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        for i, data in enumerate(dataloader):
            #data = dataset.__getitem__(i)
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data
            outputs, targets = self.get_outputs(data)

            #if type(data_id) is tuple:
            #    id = data_id[0]
            #    slice = 0
            #else:
            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                _, id, slice = data_id[0].split('_')

            #print(id)
            if id != patient_id and patient_id != None:
                loss_log[patient_id] = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0).item()
                print(patient_id + ", " + str(loss_log[patient_id]))
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

        loss_log[patient_id] = 1 - loss_f(patient_3d_image, patient_3d_label, smooth = 0).item()
        print(patient_id + ", " + str(loss_log[patient_id]))
        print("Average Dice Loss: " + str(sum(loss_log.values())/len(loss_log)))

        self.exp.set_model_state('train')
        return loss_log