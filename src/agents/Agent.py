from src.utils.helper import saveNiiGz

import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image
from src.losses.LossFunctions import DiceLoss
import seaborn as sns
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
import math

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
        # If stacked NCAs
        if isinstance(self.model, list):
            self.optimizer = []
            self.scheduler = []
            for m in range(len(self.model)):
                self.optimizer.append(optim.Adam(self.model[m].parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas')))
                self.scheduler.append(optim.lr_scheduler.ExponentialLR(self.optimizer[m], self.exp.get_from_config('lr_gamma')))
        else:
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

    def get_outputs(self, data, **kwargs):
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
        loss = 0
        loss_ret = {}
        #print(outputs.shape)
        if len(outputs.shape) == 5:
            for m in range(outputs.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[...])
                #if m == 0:
                #    loss_loc = loss_loc * 100
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(outputs.shape[-1]):
                if 1 in targets[..., m]:
                    loss_loc = loss_f(outputs[..., m], targets[..., m])
                    #if m == 0:
                    #    loss_loc = loss_loc * 100
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return loss_ret

    def intermediate_results(self, epoch, loss_log):
        r"""Write intermediate results to tensorboard
            Args:
                epoch (int): Current epoch
                los_log ([loss]): Array of losses
        """
        for key in loss_log.keys():
            #print(loss_log)
            #print(sum(loss_log[key]))
            #print(len(loss_log[key]))
            if len(loss_log[key]) != 0:
                average_loss = sum(loss_log[key]) / len(loss_log[key])
            else:
                average_loss = 0
            print(epoch, "loss =", average_loss)
            self.exp.write_scalar('Loss/train/' + str(key), average_loss, epoch)

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
        for key in loss_log.keys():
            img_plot = self.plot_results_byPatient(loss_log[key])
            self.exp.write_figure('Patient/dice/mask' + str(key), img_plot, epoch)
            if len(loss_log[key]) > 0:
                self.exp.write_scalar('Dice/test/mask' + str(key), sum(loss_log[key].values())/len(loss_log[key]), epoch)
                self.exp.write_histogram('Dice/test/byPatient/mask' + str(key), np.fromiter(loss_log[key].values(), dtype=float), epoch)
        param_lst = []
        # ADD AGAIN TODO
        #for param in self.model.parameters():
        #    param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)

    def getAverageDiceScore(self, useSigmoid=True, tag = ""):
        r"""Get the average Dice test score.
            Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=[])

        return loss_log

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
            loss_log = {}
            for m in range(self.output_channels):
                loss_log[m] = []
            self.initialize_epoch()
            print('Dataset size: ' + str(len(dataloader)))
            for i, data in enumerate(dataloader):
                loss_item = self.batch_step(data, loss_f)
                for key in loss_item.keys():
                    loss_log[key].append(loss_item[key])
            self.intermediate_results(epoch, loss_log)
            if epoch % self.exp.get_from_config('evaluate_interval') == 0:
                print("Evaluate model")
                self.intermediate_evaluation(dataloader, epoch)
            #if epoch % self.exp.get_from_config('ood_interval') == 0:
            #    print("Evaluate model in OOD cases")
            #    self.ood_evaluation(epoch=epoch)
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

    def ood_evaluation(self, ood_cases=["random_noise", "random_spike", "random_anitrosopy"], epoch=0):
        print("OOD EVALUATION")
        dataset_train = self.exp.dataset
        diceLoss = DiceLoss(useSigmoid=True)
        for augmentation in ood_cases:
            dataset_eval = Nii_Gz_Dataset_lowPass(aug_type=augmentation)
            self.exp.dataset = dataset_eval
            loss_log = self.test(diceLoss, tag='ood/' + str(augmentation) + '/')
            #img_plot = self.plot_results_byPatient(loss_log)
            #self.exp.write_figure('Patient/dice', img_plot, epoch)
            for key in loss_log.keys():
                self.exp.write_scalar('ood/Dice/' + str(key) + ", " + str(augmentation), sum(loss_log[key].values())/len(loss_log[key]), epoch)
                self.exp.write_histogram('ood/Dice/' + str(key) + ", " + str(augmentation) + '/byPatient', np.fromiter(loss_log[key].values(), dtype=float), epoch)
        self.exp.dataset = dataset_train


    def test(self, loss_f, save_img = None, tag='test/img/', **kwargs):
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
        patient_real_Img = None
        loss_log = {}
        for m in range(self.output_channels):
            loss_log[m] = {}

        if save_img == None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        for i, data in enumerate(dataloader):
            #data = dataset.__getitem__(i)
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data
            outputs, targets = self.get_outputs(data, full_img=True)

            #if type(data_id) is tuple:
            #    id = data_id[0]
            #    slice = 0
            #else:
            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                text = data_id[0].split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None

            #print(id)

            # --------------- 2D ---------------------
            if dataset.slice is not None:
                # Calculate Dice
                if id != patient_id and patient_id != None:
                    out = patient_id + ", "
                    for m in range(patient_3d_image.shape[3]):
                        if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                            loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() #,, mask = patient_3d_label[...,4].bool()

                            if math.isnan(loss_log[m][patient_id]):
                                loss_log[m][patient_id] = 0

                            # save img
                            #label_out = torch.sigmoid(patient_3d_image[..., 0])
                            #label_out[label_out < 0.5] = 0
                            #label_out[label_out > 0.5] = 1
                            #nib_save = nib.Nifti1Image(label_out  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                            #nib.save(nib_save, os.path.join("/home/jkalkhof_locale/Documents/Data/Prostate/Hippocampus/UNet/", str(len(loss_log[0])) + ".nii.gz"))

                            #nib_save = nib.Nifti1Image(torch.sigmoid(patient_real_Img[..., 0])  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                            #nib.save(nib_save, os.path.join("/home/jkalkhof_locale/Documents/Data/Prostate/Hippocampus/UNet/", str(len(loss_log[0])) + "_real.nii.gz"))

                            #nib_save = nib.Nifti1Image(patient_3d_label[..., 0]  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                            #nib.save(nib_save, os.path.join("/home/jkalkhof_locale/Documents/Data/Prostate/Hippocampus/UNet/", str(len(loss_log[0])) + "_ground.nii.gz"))

                            #print(loss_log[m])
                            #print(patient_id + ", " + str(m) + ", " + str(loss_log[m][patient_id]))
                            out = out + str(loss_log[m][patient_id]) + ", "
                        else:
                            out = out + " , "
                    print(out)
                    patient_id, patient_3d_image, patient_3d_label = id, None, None

                if patient_3d_image == None:
                    patient_id = id
                    patient_3d_image = outputs.detach().cpu()#[:, :, :, 0] #:4
                    patient_3d_label = targets.detach().cpu()#[:, :, :, 0]
                    patient_real_Img = inputs.detach().cpu()
                else:
                    patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))#[:, :, :, 0])) #:4
                    patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))#[:, :, :, 0])) #:4
                    patient_real_Img = torch.vstack((patient_real_Img, inputs.detach().cpu()))#[:, :, :, 0])) #:4
                # Add image to tensorboard
                if i in save_img: #np.random.random() < chance:
                    self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                    convert_image(self.prepare_image_for_display(inputs.detach().cpu()).numpy(), 
                    self.prepare_image_for_display(outputs.detach().cpu()).numpy(), 
                    self.prepare_image_for_display(targets.detach().cpu()).numpy(), 
                    encode_image=False), self.exp.currentStep)
            # --------------------------------- 3D ----------------------------
            else: 
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_3d_real_Img = inputs.detach().cpu()
                patient_id = id

                print(patient_3d_image.shape)
                print(patient_3d_label.shape)

                loss_log[0][patient_id] = 1 - loss_f(patient_3d_image[...,0], patient_3d_label, smooth = 0).item()
                
                # Add image to tensorboard
                print(patient_3d_image.shape)
                if i in save_img and True: #np.random.random() < chance:
                    
                    self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                    convert_image(self.prepare_image_for_display(patient_3d_real_Img[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    self.prepare_image_for_display(patient_3d_image[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    self.prepare_image_for_display(patient_3d_label[:,:,:,5:6,:].detach().cpu()).numpy(), 
                    encode_image=False), self.exp.currentStep)

                #if math.isnan(loss_log[m][patient_id]):
                #    loss_log[m][patient_id] = 0

            if dataset.slice is not None:
                out = patient_id + ", "
                for m in range(patient_3d_image.shape[3]):
                    if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() # ,mask = patient_3d_label[...,4].bool()
                        out = out + str(loss_log[m][patient_id]) + ", "
                    else:
                        out = out + " , "
                print(out)
            for key in loss_log.keys():
                if len(loss_log[key]) > 0:
                    print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))


        self.exp.set_model_state('train')
        return loss_log