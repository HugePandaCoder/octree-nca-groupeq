import torch
import numpy as np
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import random
import torchio as tio
from matplotlib import pyplot as plt
from src.models.Model_Preprocess import PreprocessNCA
import torch.optim as optim
from itertools import chain
from src.losses.LossFunctions import DiceFocalLoss_2
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torchvision.models import vgg19

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=0.1):
        """
        Initializes the Huber loss function.
        
        Parameters:
        - delta: The threshold at which the loss transitions from quadratic to linear.
        """
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        """
        Calculates the Huber loss between `input` and `target`.
        
        Parameters:
        - input: Predicted images.
        - target: Ground truth images.
        """
        abs_diff = torch.abs(input - target)
        quadratic = torch.where(abs_diff <= self.delta, 0.5 * abs_diff ** 2, self.delta * (abs_diff - 0.5 * self.delta))
        return quadratic.mean()

class CustomLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-9, scale=1.0):
        """
        Custom loss that penalizes outputs near zero more heavily.
        :param epsilon: Small constant to avoid division by zero.
        :param scale: Scaling factor to adjust the steepness of the penalty curve.
        """
        super(CustomLoss, self).__init__()
        self.epsilon = epsilon
        self.scale = scale

    def forward(self, outputs):
        """
        Compute the custom loss.
        :param outputs: The predictions from the model.
        :param targets: The ground truth values.
        """
        # Ensure outputs are not exactly zero by adding a small constant (epsilon)
        safe_outputs = outputs + self.epsilon
        
        # Calculate the inverse loss, heavily penalizing values close to zero
        loss = self.scale / torch.abs(safe_outputs)
        
        # Optionally, combine with a standard loss (e.g., MSE or L1) to ensure learning
        # standard_loss = torch.nn.functional.mse_loss(outputs, targets)
        # total_loss = loss + standard_loss
        
        # Here, we return the custom loss directly
        return torch.mean(loss)

class ParameterRegularizationLoss:
    def __init__(self, model):
        # Store the original parameters
        self.original_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def compute_loss(self, model, lambda_reg=0.01):
        # Compute the regularization loss as the sum of squared differences
        reg_loss = 0.0
        for name, param in model.named_parameters():
            reg_loss += (param - self.original_params[name]).pow(2).sum()
        reg_loss *= lambda_reg
        return reg_loss

class EWC(object):
    def __init__(self, model, dataloader, device, agent):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p, device=device) for n, p in self.params.items()}
        self.optimal_params = {n: p.clone().detach() for n, p in self.params.items()}

        self.compute_fisher_matrix(agent)

    def compute_fisher_matrix(self, agent):
        self.model.eval()
        for _, data in enumerate(self.dataloader):
            self.model.zero_grad()
            #inputs, targets = inputs.to(self.device), targets.to(self.device)
            #outputs = self.model(inputs)
            data = agent.prepare_data(data)
            outputs, targets = agent.get_outputs(data, return_channels=False)
            
            loss = outputs.log().mul(targets).sum()  # Assuming targets are one-hot encoded
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.fisher[n] += p.grad ** 2 / len(self.dataloader)

    def ewc_loss(self, lambda_ewc):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss += (lambda_ewc / 2) * (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss

class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(device) # Using VGG up to layer 36 for feature extraction
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input_img, target_img):
        # Normalize images as required by pretrained VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(input_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(input_img.device)
        input_img_normalized = (input_img - mean) / std
        target_img_normalized = (target_img - mean) / std

        input_features = self.vgg(input_img_normalized)
        target_features = self.vgg(target_img_normalized)

        # Calculate perceptual loss
        loss = F.l1_loss(input_features, target_features)
        return loss

class Agent_Med_NCA_finetuning(MedNCAAgent):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """

    def initialize(self):
        # create test  model
        super().initialize()
        self.preprocess_model = PreprocessNCA(channel_n=16, fire_rate=0.3, device=self.device, hidden_size=256).to(self.device)
        self.preprocess_model2 = PreprocessNCA(channel_n=16, fire_rate=0.3, device=self.device, hidden_size=256).to(self.device)
        self.optimizer_test = optim.Adam(chain(self.preprocess_model.parameters(), self.preprocess_model2.parameters()), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler_test = optim.lr_scheduler.ExponentialLR(self.optimizer_test, self.exp.get_from_config('lr_gamma'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
        self.ewc = None
        self.reg_loss_calculator = None

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        

        if self.model.training:
            inputs, targets, inputs_loc = self.model(inputs, targets, return_channels=True, preprocess_model = (self.preprocess_model, self.preprocess_model2))
            return inputs, targets, inputs_loc
        else:
            inputs, targets = self.model(inputs, targets, return_channels=False, preprocess_model = (self.preprocess_model, self.preprocess_model2))
            return inputs, targets

    def per_image_normalized_tv_loss(img):
        """
        Compute the Total Variation Loss independently per image in the batch and then normalize.
        img: Tensor of shape (B, C, H, W)
        """
        # Initialize TV loss
        tv_loss = 0.0
        
        # Loop over the batch
        for i in range(img.shape[0]):
            # Calculate the difference of neighboring pixel-values for each image
            diff_i = torch.abs(img[i, :, 1:, :] - img[i, :, :-1, :])
            diff_j = torch.abs(img[i, :, :, 1:] - img[i, :, :, :-1])
            
            # Summing the differences in both directions and normalizing by the number of elements
            tv_loss_per_image = (torch.sum(diff_i) + torch.sum(diff_j)) / img[i].numel()
            
            # Accumulate the TV loss from all images
            tv_loss += tv_loss_per_image
        
        # Average the TV loss across the batch
        tv_loss /= img.shape[0]
        return tv_loss

    def normalized_total_variation_loss(self, img):
        """
        Compute the Normalized Total Variation Loss.
        img: Tensor of shape (B, C, H, W)
        """
        # Calculate the difference of neighboring pixel-values
        diff_i = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        diff_j = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        
        # Summing the differences in both directions and normalizing
        tv_loss = (torch.sum(diff_i) + torch.sum(diff_j)) / img.numel()
        return tv_loss

    def preprocess_grayscale_for_vgg(self, images):
        """
        Convert a batch of grayscale images to the format expected by VGG.
        images: Tensor of shape (B, H, W, C) where C=1
        Returns: Tensor of shape (B, C, H, W) with C=3
        """
        images = images.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        images_rgb = images.repeat(1, 3, 1, 1)  # Repeat the grayscale channel to mimic RGB
        return images_rgb

    def gaussian_window(self, size, sigma):
        """
        Generates a 2D Gaussian window.
        """
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = coords**2
        g = (-g / (2 * sigma**2)).exp()
        g /= g.sum()
        return g.outer(g)

    def ssim(self, img1, img2, window_size=11, window_sigma=1.5, size_average=True):
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        """
        channel = img1.size(1)
        window = self.gaussian_window(window_size, window_sigma).to(img1.device)
        window = window.expand(channel, 1, window_size, window_size)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """

        #if self.ewc is None:
        #    data_loader = torch.utils.data.DataLoader(self.exp.dataset, shuffle=True, batch_size=self.exp.get_from_config('batch_size'))
        #    self.ewc = EWC(self.model, data_loader, self.device, self)
        #    self.lambda_ewc = 500

        if self.reg_loss_calculator is None:
            self.reg_loss_calculator = ParameterRegularizationLoss(self.model)

        data = self.prepare_data(data)
        #rnd = random.randint(0, 1000000000)
        #random.seed(rnd)
        outputs, targets, inputs_loc = self.get_outputs(data, return_channels=False)

        #plt.imshow(targets[0, :, :, 0].detach().cpu().numpy())
        #plt.show()

        #plt.imshow(((inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy()+1)/2)
        #plt.imshow((inputs_loc_3_ori[0, :, :, 0:1].detach().cpu().numpy()+1)/2)
        #plt.show()
        if False:
            self.exp.write_img('before',
                                    inputs_loc[1][0, :, :, 0:1].detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            self.exp.write_img('difference',
                                    torch.abs(inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]).detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            self.exp.write_img('after',
                                    inputs_loc[0][0, :, :, 0:1].detach().cpu().numpy(),
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
        else:
            difference = inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]
            difference = (difference - difference.min()) / (difference.max() - difference.min())
            cat_img =  torch.cat((inputs_loc[1][0, :, :, 0:1], difference, inputs_loc[0][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
            self.exp.write_img('preprocessing_main_level',
                                    cat_img,
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
            difference = inputs_loc[3][0, :, :, 0:1] - inputs_loc[2][0, :, :, 0:1]
            difference = (difference - difference.min()) / (difference.max() - difference.min())
            cat_img =  torch.cat((inputs_loc[3][0, :, :, 0:1], difference, inputs_loc[2][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
            self.exp.write_img('preprocessing_patch_level',
                                    cat_img,
                                    #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                    self.exp.currentStep)
        #random.seed(rnd)
        #outputs2, targets, inputs_loc_2 = self.get_outputs(data, return_channels=True)

        #plt.imshow(targets[0, :, :, 0].detach().cpu().numpy())
        #plt.show()

        self.optimizer_test.zero_grad()
        loss = 0
        #print(outputs.shape, targets.shape)
        #if len(outputs.shape) == 5:
        #    for m in range(targets.shape[-1]):
        #        loss_loc = loss_f(outputs[..., m], targets[...])
        #        loss = loss + loss_loc
        #        loss_ret[m] = loss_loc.item()
        #else:
        #    for m in range(targets.shape[-1]):
        #        if 1 in targets[..., m]:
        #            loss_loc = loss_f(outputs[..., m], targets[..., m])
        #            loss = loss + loss_loc
        #            loss_ret[m] = loss_loc.item()


        if False:
            mse = torch.nn.MSELoss()
            l1 = torch.nn.L1Loss()
            #loss = mse(torch.sum(torch.sigmoid(outputs)), torch.sum(torch.sigmoid(outputs2)))
            #target = torch.sigmoid(outputs2)
            #target[target > 0.5] = 1
            #target[target < 0.5] = 0

            #print("INPUTS LOC ", inputs_loc[0].shape)
            def toFourier(tensor):
                tensor = tensor.transpose(1, 3)
                tensor_loc = torch.fft.fft2(tensor, s = (12, 12), norm="forward")
                tensor = torch.fft.ifft2(tensor_loc, norm="forward", s=(tensor.shape[2], tensor.shape[3]))
                #x_start, x_end = tensor.shape[2]//2 -8, tensor.shape[2]//2 + 8
                #y_start, y_end = tensor.shape[3]//2 -8, tensor.shape[3]//2 + 8
                #tensor = torch.fft.fftshift(tensor)#[..., x_start:x_end, y_start:y_end]
                return tensor.real

            # >>>>> Loss MSE-variance between outputs 
            max_val = torch.max(torch.abs(inputs_loc[6][..., self.input_channels:]))
            loss = mse(inputs_loc[6][..., self.input_channels:] / max_val, inputs_loc[7][..., self.input_channels:] / max_val)
            
            # >>>>>Loss MSE-variance between mid layers
            max_val = torch.max(torch.abs(inputs_loc[4][..., self.input_channels:]))
            loss2 = mse(inputs_loc[4][..., self.input_channels:] / max_val, inputs_loc[5][..., self.input_channels:] / max_val)
                #l1(torch.log(torch.clamp(inputs_loc[2][..., self.input_channels:]*-1, 1e-10)), torch.log(torch.clamp(inputs_loc[3][..., self.input_channels:]*-1, 1e-10))))
            
            #loss2 = l1(inputs_loc[4][..., self.input_channels:], inputs_loc[5][..., self.input_channels:])
            
            dice_f = DiceFocalLoss_2()

            #dice_loss = l1(torch.sigmoid(inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]), torch.sigmoid(inputs_loc[7][..., self.input_channels:self.input_channels+self.output_channels])) + \
            #    l1(torch.sigmoid(inputs_loc[4][..., self.input_channels:self.input_channels+self.output_channels]), torch.sigmoid(inputs_loc[5][..., self.input_channels:self.input_channels+self.output_channels]))

            dice_loss = dice_f((inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]), (inputs_loc[7][..., self.input_channels:self.input_channels+self.output_channels])) + \
                dice_f((inputs_loc[4][..., self.input_channels:self.input_channels+self.output_channels]), (inputs_loc[5][..., self.input_channels:self.input_channels+self.output_channels]))

            loss_kd = F.cross_entropy(F.softmax(inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels], dim=3).float(), F.softmax(inputs_loc[7][..., self.input_channels:self.input_channels+self.output_channels], dim=3).float())  + \
                F.cross_entropy(F.softmax(inputs_loc[4][..., self.input_channels:self.input_channels+self.output_channels], dim=3).float(), F.softmax(inputs_loc[5][..., self.input_channels:self.input_channels+self.output_channels], dim=3).float())

            # >>>>> Loss L1 between fourier
            #loss3 = (mse(toFourier(inputs_loc[0][..., 0:self.input_channels]), toFourier(inputs_loc[1][..., 0:self.input_channels])) + \
            #    mse(toFourier(inputs_loc[2][..., 0:self.input_channels]), toFourier(inputs_loc[3][..., 0:self.input_channels])))
            
            # >>>>> Loss mse between fourier
            #loss4 = (mse(apply_gaussian_blur(inputs_loc[0][..., 0:self.input_channels]), apply_gaussian_blur(inputs_loc[1][..., 0:self.input_channels])) + \
            #    mse(apply_gaussian_blur(inputs_loc[2][..., 0:self.input_channels]), apply_gaussian_blur(inputs_loc[3][..., 0:self.input_channels])))
            loss5 = (l1(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]) + \
                        l1(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels]))
            
            loss6 = (mse(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]) + \
                        mse(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels]))

            #p_loss = PerceptualLoss(self.device)        

            #perceptual_loss = p_loss(self.preprocess_grayscale_for_vgg(inputs_loc[0][..., 0:self.input_channels]), self.preprocess_grayscale_for_vgg(inputs_loc[1][..., 0:self.input_channels])) + \
            #    p_loss(self.preprocess_grayscale_for_vgg(inputs_loc[2][..., 0:self.input_channels]), self.preprocess_grayscale_for_vgg(inputs_loc[3][..., 0:self.input_channels]))

            ssim_loss = (1-self.ssim(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]))+1 + \
                (1-self.ssim(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels]))+1


            print(inputs_loc[0][..., 0:self.input_channels].shape)

            #loss_total_var = torch.abs(self.per_image_normalized_tv_loss(inputs_loc[0][..., 0:self.input_channels]) - self.normalized_total_variation_loss(inputs_loc[1][..., 0:self.input_channels])) + \
            #      torch.abs(self.normalized_total_variation_loss(inputs_loc[2][..., 0:self.input_channels]) - self.normalized_total_variation_loss(inputs_loc[3][..., 0:self.input_channels]))
            loss_total_var = self.normalized_total_variation_loss(inputs_loc[0][..., 0:self.input_channels]) + \
                self.normalized_total_variation_loss(inputs_loc[2][..., 0:self.input_channels])
                            

            # NQM loss
            nqm_loss = 0
            nqm_loss2 = 0
            for b in range(inputs_loc[4].shape[0]):
                stack = torch.stack([inputs_loc[4][b, ..., self.input_channels:self.input_channels+self.output_channels], inputs_loc[5][b, ..., self.input_channels:self.input_channels+self.output_channels]], dim=0)
                outputs = torch.mean(stack, dim=0)
                nqm_loss += self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())

                stack = torch.stack([inputs_loc[6][b, ..., self.input_channels:self.input_channels+self.output_channels], inputs_loc[7][b, ..., self.input_channels:self.input_channels+self.output_channels]], dim=0)
                outputs = torch.mean(stack, dim=0)
                nqm_loss2 += self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())

            nqm_loss = nqm_loss / inputs_loc[4].shape[0]
            nqm_loss2 = nqm_loss2 / inputs_loc[4].shape[0]

            seg_something_loss = torch.mean(torch.sigmoid(inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]))

            criterion = CustomLoss(epsilon=1e-9, scale=0.0001)

            loss_ret = {}
            loss = ((nqm_loss + nqm_loss2)/2 + (loss5 +  loss6)*5 + criterion(seg_something_loss)) #(ssim_loss/2 + loss5 + loss6)/4)# (loss5+loss6)/3)#*800 + loss5#loss3 #loss4 +  loss4*5 +   + loss_total_var/5
            print(nqm_loss.item(), nqm_loss2.item(), loss5.item(), loss6.item(), loss.item())#, loss5.item())
            loss_ret[0] = loss.item()
            #loss_ret[1] = loss2.item()
            #loss_ret[2] = loss3.item()

            

            weight_sum = 0
            for param in self.preprocess_model.parameters():
                weight_sum += param.data.sum()

            print("PARAM SUM", weight_sum)

            if loss != 0:
                loss.backward()

                self.optimizer_test.step()
                self.scheduler_test.step()
        
        else:
            dice_f = DiceFocalLoss_2()

            #dice_loss = dice_f((inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]), (inputs_loc[7][..., self.input_channels:self.input_channels+self.output_channels])) + \
            #    dice_f((inputs_loc[4][..., self.input_channels:self.input_channels+self.output_channels]), (inputs_loc[5][..., self.input_channels:self.input_channels+self.output_channels]))

            #ewc_loss = self.ewc.ewc_loss(self.lambda_ewc)
            
            # NQM loss
            nqm_loss = 0
            nqm_loss2 = 0
            for b in range(inputs_loc[4].shape[0]):
                stack = torch.stack([inputs_loc[4][b:b+1, ..., self.input_channels:self.input_channels+self.output_channels], inputs_loc[5][b:b+1, ..., self.input_channels:self.input_channels+self.output_channels]], dim=0)
                outputs = torch.mean(stack, dim=0)

                nq_loc = self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
                nqm_loss += nq_loc #* nq_loc

                stack = torch.stack([inputs_loc[6][b:b+1, ..., self.input_channels:self.input_channels+self.output_channels], inputs_loc[7][b:b+1, ..., self.input_channels:self.input_channels+self.output_channels]], dim=0)
                outputs = torch.mean(stack, dim=0)
                nq_loc = self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy())
                nqm_loss2 += nq_loc #* nq_loc

            nqm_loss = nqm_loss / inputs_loc[4].shape[0]
            nqm_loss2 = nqm_loss2 / inputs_loc[4].shape[0]

            reg_loss = self.reg_loss_calculator.compute_loss(self.model, lambda_reg=100)

            seg_something_loss = torch.mean(torch.sigmoid(inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]))

            #criterion = CustomLoss(epsilon=1e-9, scale=0.0001)
            criterion = CustomLoss(epsilon=1e-9, scale=0.0001)

            #ssim_loss = (1-self.ssim(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]))+1 + \
            #    (1-self.ssim(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels]))+1

            #print(ssim_loss.item())

            huber_loss = HuberLoss(delta=200) 

            hl_loss = huber_loss(inputs_loc[0][..., 0:self.input_channels], inputs_loc[1][..., 0:self.input_channels]) + \
                huber_loss(inputs_loc[2][..., 0:self.input_channels], inputs_loc[3][..., 0:self.input_channels])

            loss_ret = {}# 
            loss = nqm_loss*6 +  reg_loss  + criterion(seg_something_loss) # + nqm_loss2*10 + hl_loss*3000)/50#seg_something_loss*0.1  + (ssim_loss/5)
            print(nqm_loss.item(), nqm_loss2.item(), reg_loss.item(), criterion(seg_something_loss).item(), hl_loss.item(), loss.item())#, loss5.item())
            loss_ret[0] = loss.item()

            if loss != 0:
                loss.backward()

                learning_rates = [param_group['lr'] for param_group in self.optimizer.param_groups]
                print("Learning rates:", learning_rates)

                self.optimizer.step()
                self.scheduler.step()

                #self.optimizer_test.step()
                #self.scheduler_test.step()

        return loss_ret
    
    def labelVariance(self, images: torch.Tensor, median: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = np.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = np.sqrt(stdd)

        #plt.imshow(stdd[0, :, :, 0])
        #plt.show()
        #exit()

        print("NQM Score: ", np.sum(stdd) / (np.sum(median)+1e-9))

        return np.sum(stdd) / (np.sum(median)+1e-9)


def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel with PyTorch."""
    coords = torch.arange(size).float()
    coords -= size // 2

    g = coords ** 2
    g = (-g / (2 * sigma ** 2)).exp()

    g /= g.sum()
    gaussian_kernel = torch.outer(g, g)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]

    return gaussian_kernel

def apply_gaussian_blur(image, kernel_size=13, sigma=2):
    """Applies Gaussian blur to an image using convolution."""
    channels = image.shape[1]
    gaussian_kernel_weights = gaussian_kernel(kernel_size, sigma).to(image.device)
    gaussian_kernel_weights = gaussian_kernel_weights.expand(channels, 1, kernel_size, kernel_size)
    
    # Ensuring the image has the channel in the second dimension, expected shape: [N, C, H, W]
    blurred_image = F.conv2d(image, gaussian_kernel_weights, padding=kernel_size//2, groups=channels)
    return blurred_image