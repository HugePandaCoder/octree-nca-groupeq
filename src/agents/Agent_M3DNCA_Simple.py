import torch
from src.agents.Agent_UNet import UNetAgent
import torch.nn.functional as F
import random
from src.agents.Agent_MedSeg3D import Agent_MedSeg3D

class M3DNCAAgent(UNetAgent):
    """Base agent for training UNet models
    """
    def initialize(self):
        super().initialize()

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        
        inputs = inputs.permute(0, 2, 3, 4, 1)

        inputs, targets = self.model(inputs, targets)
        return inputs, targets 
        #if len(inputs.shape) == 4:
        #    return (self.model(inputs)).permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        #else:
        #    return (self.model(inputs)).permute(0, 2, 3, 4, 1), targets #.permute(0, 2, 3, 4, 1)

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        rnd = random.randint(0, 1000000000)
        random.seed(rnd)
        outputs, targets = self.get_outputs(data)
        #print("______________________")
        random.seed(rnd)
        outputs2, targets2 = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        if len(outputs.shape) == 5 and targets.shape[-1] == 1:
            for m in range(targets.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[...])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(targets.shape[-1]):
                if 1 in targets[..., m]:
                    loss_loc = loss_f(outputs[..., m], targets[..., m])
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        # loss variance
        #if len(outputs.shape) == 5:
        #    for m in range(targets.shape[-1]):
        #        loss_loc = loss_f(outputs2[..., m], targets2[...])
        #        loss = loss + loss_loc
        #        loss_ret[m] = loss_loc.item()

        # CALC NQM
        if False:
            stack = torch.stack([outputs, outputs2], dim=0)
            outputs = torch.sigmoid(torch.mean(stack, dim=0))
            stack = torch.sigmoid(stack)
            if torch.sum(stack) != 0:
                mean = torch.sum(stack, axis=0) / stack.shape[0]
                stdd = torch.zeros(mean.shape).to(self.device)
                for id in range(stack.shape[0]):
                    img = stack[id] - mean
                    img = torch.pow(img, 2)
                    stdd = stdd + img
                stdd = stdd / stack.shape[0]
                stdd = torch.sqrt(stdd)

                print("STDD", torch.min(stdd), torch.max(stdd), torch.sum(outputs))

                if torch.min(stdd) > 0:
                    nqm = torch.sum(stdd) / torch.sum(outputs)

                    if nqm > 0:
                        print("NQM: ", nqm)
                        loss = loss + nqm #
        else:
            loss += F.mse_loss(outputs, outputs2)

            #print(nqm)

        if loss != 0:
            loss.backward()

            if gradient_norm:
                print("GRADIENT NORM")
                max_norm = 1.0
                # Gradient normalization
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)


            self.optimizer.step()
            self.scheduler.step()
        return loss_ret

    #def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
    #    return super().batch_step(data, loss_f, gradient_norm)
    