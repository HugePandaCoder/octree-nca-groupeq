import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torchio as tio
import random
from matplotlib import pyplot as plt

class MedNCA_finetune(nn.Module):
    r"""Implementation of the backbone NCA of Med-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, batch_duplication: int = 1):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(MedNCA_finetune, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.batch_duplication = batch_duplication

        self.do_preprocess = False

        self.backbone_lowres = BackboneNCA(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)
        self.backbone_highres = BackboneNCA(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)

    def make_seed(self, x):
        seed = torch.zeros((x.shape[0], self.channel_n, x.shape[2], x.shape[3]), dtype=torch.float32, device=self.device)
        seed[:, :x.shape[self.input_channels], :, :] = x
        return seed

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, variance = None, pred = None, return_channels : bool = False, preprocess_model = None):
        x = self.make_seed(x).to(self.device)
        x = x.transpose(1,3)
        y = y.transpose(1,3)
        y = y.to(self.device)

        if self.training:
            if self.batch_duplication != 1:
                x = torch.cat([x] * self.batch_duplication, dim=0)
                y = torch.cat([y] * self.batch_duplication, dim=0)

            x, y, variance, pred, inputs_loc = self.forward_train(x, y, variance, pred, return_channels=return_channels, preprocess_model=preprocess_model)
            return x, y, variance, pred, inputs_loc
            
        else:
            x = self.forward_eval(x, preprocess_model=preprocess_model)
            return x, y

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, variance: torch.Tensor, pred: torch.Tensor, return_channels : bool = False, preprocess_model = None):
        down_scaled_size = (x.shape[1] // 4, x.shape[2] // 4)
        inputs_loc = self.resize4d(x.cpu(), size=down_scaled_size).to(self.device) 
        targets_loc = self.resize4d(y.cpu(), size=down_scaled_size).to(self.device)
        #variance_loc = self.resize4d(variance.cpu(), size=down_scaled_size).to(self.device) 
        #pred_loc = self.resize4d(pred.cpu(), size=down_scaled_size).to(self.device)

        # Start with low res lvl and go to high res level
        for m in range(2):
            if m == 1:
                inputs_loc_3 = inputs_loc.clone()
                inputs_loc_3[..., self.input_channels:][inputs_loc_3[..., self.input_channels:] != 0] = 0
                inputs_loc_3_ori = inputs_loc.clone()

                if self.do_preprocess:
                    inputs_loc_3[..., 0:self.input_channels] = preprocess_model[0](inputs_loc_3.clone())[..., :self.input_channels]
                    inputs_loc[..., 0:self.input_channels] = inputs_loc_3[..., :self.input_channels]

                #plt.imshow((inputs_loc[0, :, :, 0:3].detach().cpu().numpy()))
                #plt.show()

                outputs = self.backbone_highres(inputs_loc, 
                                               steps=self.steps, 
                                               fire_rate=self.fire_rate)
                # Variance training only
                outputs2_patch = None
                outputs2_patch = self.backbone_lowres(inputs_loc, 
                                                steps=self.steps, 
                                                fire_rate=self.fire_rate)
                
                outputs3_patch = None
                outputs4_patch = None
                
                # Variance training only
                # outputs3_patch = self.backbone_lowres(inputs_loc, 
                #                                 steps=self.steps, 
                #                                 fire_rate=self.fire_rate)
                
                # outputs4_patch = self.backbone_lowres(inputs_loc, 
                #                                 steps=self.steps, 
                #                                 fire_rate=self.fire_rate)
            else:
                # Preprocess Inputs
                inputs_loc_2 = inputs_loc.clone()
                inputs_loc_2_ori = inputs_loc.clone()

                if self.do_preprocess:
                    inputs_loc_2[..., 0:self.input_channels] = preprocess_model[1](inputs_loc_2.clone())[..., :self.input_channels]
                    inputs_loc[..., 0:self.input_channels] = inputs_loc_2[..., :self.input_channels]



                outputs = self.backbone_lowres(inputs_loc, 
                                                steps=self.steps, 
                                                fire_rate=self.fire_rate)
                
                # Variance training only
                outputs2_full = None
                outputs2_full = self.backbone_lowres(inputs_loc, 
                                                steps=self.steps, 
                                                fire_rate=self.fire_rate)
                
                outputs3_full = None
                outputs4_full = None

                # outputs3_full = self.backbone_lowres(inputs_loc, 
                #                                 steps=self.steps, 
                #                                 fire_rate=self.fire_rate)
                
                # outputs4_full = self.backbone_lowres(inputs_loc, 
                #                                 steps=self.steps, 
                #                                 fire_rate=self.fire_rate)

                before_patch = outputs.clone()

                # Upscale lowres features to high res
                up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                outputs = torch.permute(outputs, (0, 3, 1, 2))
                outputs = up(outputs)
                inputs_loc = x     
                outputs = torch.permute(outputs, (0, 2, 3, 1))
                # Concat lowres features with high res image             
                inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                targets_loc = y
                variance_loc = variance
                pred_loc = pred

                #print("LOC", inputs_loc.shape, targets_loc.shape, pred_loc.shape)
                #plt.imshow(inputs_loc.detach().cpu().numpy()[0, :, :, 0])
                #plt.show()
                #plt.imshow(targets_loc.detach().cpu().numpy()[0, :, :, 0])
                #plt.show()
                #plt.imshow(pred_loc.detach().cpu().numpy()[0, :, :, 0])
                #plt.show()
                #plt.imshow(variance_loc.detach().cpu().numpy()[0, :, :, 0])
                #plt.show()

                # Prepare array to store patch of 
                inputs_loc_temp = inputs_loc
                targets_loc_temp = targets_loc
                inputs_loc = torch.zeros(
                    (inputs_loc.shape[0], 
                     down_scaled_size[0], 
                     down_scaled_size[1], 
                     inputs_loc.shape[3])
                     ).to(self.device)
                targets_loc = torch.zeros(
                    (targets_loc_temp.shape[0], 
                     down_scaled_size[0], 
                     down_scaled_size[1], 
                     targets_loc_temp.shape[3])
                     ).to(self.device)
                
                # Prepare variance and pred
                variance_loc_temp = variance_loc
                pred_loc_temp = pred_loc
                variance_loc = torch.zeros(
                    (variance_loc_temp.shape[0], 
                     down_scaled_size[0], 
                     down_scaled_size[1], 
                     variance_loc_temp.shape[3])
                     ).to(self.device)
                pred_loc = torch.zeros(
                    (pred_loc_temp.shape[0], 
                     down_scaled_size[0], 
                     down_scaled_size[1], 
                     pred_loc_temp.shape[3])
                     ).to(self.device)


                # Choose random patch of upscaled image
                for b in range(inputs_loc.shape[0]): 
                    pos_x = random.randint(0, 
                                           inputs_loc_temp.shape[1] - down_scaled_size[0])
                    pos_y = random.randint(0, 
                                           inputs_loc_temp.shape[2] - down_scaled_size[1])

                    inputs_loc[b] = inputs_loc_temp[b, 
                                                    pos_x:pos_x+down_scaled_size[0], 
                                                    pos_y:pos_y+down_scaled_size[1], 
                                                    :]
                    targets_loc[b] = targets_loc_temp[b, 
                                                      pos_x:pos_x+down_scaled_size[0], 
                                                      pos_y:pos_y+down_scaled_size[1], 
                                                      :]
                    
                    # variance pred
                    variance_loc[b] = variance_loc_temp[b, 
                                                      pos_x:pos_x+down_scaled_size[0], 
                                                      pos_y:pos_y+down_scaled_size[1], 
                                                      :]
                    pred_loc[b] = pred_loc_temp[b, 
                                                      pos_x:pos_x+down_scaled_size[0], 
                                                      pos_y:pos_y+down_scaled_size[1], 
                                                      :]

        # show img
        #print("SLICE CHANGES: ", torch.sum(inputs_loc_2[0, :, :, 0:1]  - inputs_loc_2_ori[0, :, :, 0:1]))

        print("FORWARD TRAIN", outputs[..., self.input_channels:self.input_channels+self.output_channels].shape, targets_loc.shape, variance_loc.shape, pred_loc.shape)

        if return_channels:
            return outputs[..., self.input_channels+self.output_channels:], targets_loc, (inputs_loc_2, inputs_loc_2_ori, inputs_loc_3, inputs_loc_3_ori, before_patch, outputs2_full, outputs, outputs2_patch, outputs3_full, outputs3_patch, outputs4_full, outputs4_patch)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc, variance_loc, pred_loc, (inputs_loc_2, inputs_loc_2_ori, inputs_loc_3, inputs_loc_3_ori, before_patch, outputs2_full, outputs, outputs2_patch, outputs3_full, outputs3_patch, outputs4_full, outputs4_patch)
    
    def forward_eval(self, x: torch.Tensor, preprocess_model = None):
        down_scaled_size = (x.shape[1] // 4, x.shape[2] // 4)
        inputs_loc = self.resize4d(x.cpu(), size=down_scaled_size).to(self.device) 

        # Start with low res lvl and go to high res level
        for m in range(2):
            if m == 1:
                inputs_loc_3 = inputs_loc.clone()
                inputs_loc_3[..., self.input_channels:][inputs_loc_3[..., self.input_channels:] != 0] = 0
                inputs_loc_3_ori = inputs_loc.clone()

                if self.do_preprocess:
                    inputs_loc_3[..., 0:self.input_channels] = preprocess_model[0](inputs_loc_3.clone())[..., :self.input_channels]
                    inputs_loc[..., 0:self.input_channels] = inputs_loc_3[..., 0:self.input_channels]

                outputs = self.backbone_highres(inputs_loc, 
                                               steps=self.steps, 
                                               fire_rate=self.fire_rate)
            else:
                inputs_loc_2 = inputs_loc.clone()
                inputs_loc_2_ori = inputs_loc.clone()
                
                if self.do_preprocess:
                    inputs_loc_2[..., 0:self.input_channels] = preprocess_model[1](inputs_loc_2.clone())[..., :self.input_channels]
                    inputs_loc[..., 0:self.input_channels] = inputs_loc_2[..., 0:self.input_channels]

                outputs = self.backbone_lowres(inputs_loc, 
                                                steps=self.steps, 
                                                fire_rate=self.fire_rate)
                # Upscale lowres features to high res
                up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                outputs = torch.permute(outputs, (0, 3, 1, 2))
                outputs = up(outputs)   
                inputs_loc = x  
                outputs = torch.permute(outputs, (0, 2, 3, 1))       
                # Concat lowres features with high res image     
                inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels]

    def resize4d(self, img: torch.Tensor, size: tuple = (64,64), factor: int = 4, label: bool = False) -> torch.Tensor:
        r"""Resize input image
            #Args
                img: 4d image to rescale
                size: image size
                factor: scaling factor
                label: is Label?
        """
        if label:
            transform = tio.Resize((size[0], size[1], -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], -1))
        img = transform(img)
        return img