
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import torch, math, numpy as np, einops

from src.utils.helper import merge_img_label_gt_simplified

class MedNCAAgent_extrapolation(MedNCAAgent):

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        # data["image"]: BCHW
        # data["label"]: BCHW
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        #print(outputs.shape, targets.shape)
        #2D: outputs: BHWC, targets: BHWC
        loss = loss_f(outputs, targets)
        loss_ret[0] = loss.item()

        if loss != 0:
            loss.backward()

            if gradient_norm or self.exp.get_from_config('track_gradient_norm'):
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

            if gradient_norm:
                max_norm = 1.0
                # Gradient normalization

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)

            if self.exp.get_from_config('track_gradient_norm'):
                if not hasattr(self, 'epoch_grad_norm'):
                    self.epoch_grad_norm = []
                self.epoch_grad_norm.append(total_norm)

            self.optimizer.step()
            if not self.exp.get_from_config('update_lr_per_epoch'):
                self.update_lr()
            
            if self.exp.get_from_config('apply_ema') and self.exp.get_from_config('ema_update_per') == 'batch':
                self.ema.update()

        return loss_ret

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        
        inputs, targets = data['image'], data['label']
        del targets 
        targets = inputs.clone()
        #2D: inputs: BCHW, targets: BCHW

        margin = self.exp.get_from_config("extrapolation_margin")
        inputs[:, :, :margin, :] = 0
        inputs[:, :, -margin:, :] = 0
        inputs[:, :, :, :margin] = 0
        inputs[:, :, :, -margin:] = 0


        inputs, targets = self.model(inputs, targets, self.exp.get_from_config('batch_duplication'))
        #2D: inputs: BHWC, targets: BHWC
        return inputs, targets
    
    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, split='test'):
        #assert isinstance(loss_f, torch.nn.MSELoss), "loss_f must be a torch.nn.Module"
        loss_f = torch.nn.MSELoss()


        # Prepare dataset for testing
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.exp.set_model_state('test')

        # Prepare arrays
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        patient_real_Img = None
        loss_log = {}
        for m in range(self.output_channels):
            loss_log[m] = {}
        if save_img == None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]


        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            if 'name' in data:
                name = data['name']
            outputs, targets = self.get_outputs(data, full_img=True, tag="0")

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None

            
            # --------------- 2D ---------------------
            # If next patient
            if (id != patient_id or dataset.slice == -1) and patient_id != None:
                out = str(patient_id) + ", "

                loss_log[0][patient_id] = loss_f(patient_3d_image, patient_3d_label).item()
                print("PATIENT ID", patient_id, loss_log[0][patient_id])

                
                patient_id, patient_3d_image, patient_3d_label = id, None, None
            # If first slice of volume
            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_real_Img = inputs.detach().cpu()
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))
                patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))
                patient_real_Img = torch.vstack((patient_real_Img, inputs.detach().cpu()))
            # Add image to tensorboard
            
            patient_real_Img = einops.rearrange(patient_real_Img, "b c h w -> b h w c")

            if i in save_img: 
                self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                merge_img_label_gt_simplified(patient_real_Img[0:1, ...], patient_3d_image[0:1, ...], patient_3d_label[0:1, ...], dataset.is_rgb),
                                #merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                self.exp.currentStep)
                #self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                #                    merge_img_label_gt(np.squeeze(inputs.detach().cpu().numpy()), 
                #                                        torch.sigmoid(outputs).detach().cpu().numpy(), 
                #                                        targets.detach().cpu().numpy()), 
                #                    self.exp.currentStep)
        # If 2D
        out = str(patient_id) + ", "
        for m in range(patient_3d_label.shape[-1]):
            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 
                out = out + str(loss_log[m][patient_id]) + ", "
            else:
                out = out + " , "
        print(out)
        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                average = sum(loss_log[key].values())/len(loss_log[key])
                print("Average Dice Loss 3d: " + str(key) + ", " + str(average))
                print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))
                self.exp.write_scalar('Loss/test/' + str(key), average, self.exp.currentStep)
                self.exp.write_scalar('Loss/test_std/' + str(key), self.standard_deviation(loss_log[key]), self.exp.currentStep)

        self.exp.set_model_state('train')
        return loss_log