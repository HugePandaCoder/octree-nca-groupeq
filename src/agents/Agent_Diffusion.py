import numpy as np
from src.agents.Agent_NCA import Agent_NCA
import torch
import random
import torch.nn.functional as F

class Agent_Diffusion(Agent_NCA):
    def initialize(self, beta_schedule='linear'):
        super().initialize()
        self.timesteps = self.exp.get_from_config('timesteps')
        self.beta_schedule = beta_schedule
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, self.betas, self.sqrt_recip_alphas, \
            self.posterior_variance = self.calc_schedule()

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        print("X_START", torch.max(x_start), torch.min(x_start))
        print("NOISE", torch.max(noise), torch.min(noise))
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # (sqrt_alphas_cumprod_t.get_device(), x_start.get_device(), sqrt_one_minus_alphas_cumprod_t.get_device(),noise.get_device())
        #print("ALPHAS_sQRt", sqrt_alphas_cumprod_t)
        #print("ALPHAS_sQRt_onemINUS", sqrt_one_minus_alphas_cumprod_t)
        noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        #print("NOISY_IMAGE", torch.max(noisy_image), torch.min(noisy_image))
        #rmax, rmin = torch.max(noisy_image), torch.min(noisy_image)
        #noisy_image = ((noisy_image - rmin) / (rmax - rmin)) *2 -1

        return noisy_image

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2).to(self.device)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def calc_schedule_wrong(self):
        betas = torch.linspace(0, 1, self.timesteps).to(self.device)
        alphas = 1 - betas
        sqrt_alphas_cumprod = alphas
        sqrt_one_minus_alphas_cumprod = betas

        posterior_variance = betas 
        sqrt_recip_alphas = alphas

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance
    
    def calc_schedule(self):
        # define beta schedule
        betas = 0
        if self.beta_schedule == "linear":
            betas = self.linear_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "cosine":
            betas = self.cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "quadratic":
            betas = self.quadratic_beta_schedule(timesteps=self.timesteps)
        elif self.beta_schedule == "sigmoid":
            betas = self.sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            NotImplementedError()

        betas = betas.to(self.device)

        # define alphas
        alphas = 1. - betas
        print("ALPHAS", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # (alphas, axis=0)
        print("ALPHAS_CUMPROD", alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        print("ALPHAS_CUMPROD_PREV", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        print("SQRT_RECIP_ALPHAS", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        print("SQRT_ALPHAS_CUMPROD", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        print("SQRT_ONE_MINUS_ALPHAS_CUMPROD", sqrt_one_minus_alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance

    def prepare_data(self, data, t, label=None, eval=False):
        r"""
        preprocessing of data
        :param data: images
        :param t: current time steps
        :param batch_size:
        :param label:
        :return: corrupt images, associated noise
        """
        id, img, _ = data
        img = img.to(self.device)
        
        #noise = img.clone()
        
        #noise = torch.randn_like(img).to(self.device)

        if False:
            noise = torch.rand(img.shape).to(self.device)*2 -1 #orch.tensor.uniform_(noise)
            print(torch.max(noise), torch.min(noise))
            img_noisy = self.q_sample(x_start=img, t=t, noise=noise)
            img_noisy = torch.clip(img_noisy, -1, 1)
            noise = img_noisy - img
        #else:
        #    noise = torch.randn_like(img).to(self.device)
        #    noise -= noise.min(1, keepdim=True)[0]
        #    noise /= noise.max(1, keepdim=True)[0]
        #    img_noisy = self.q_sample(x_start=img, t=t, noise=noise)

        noise, img_noisy = self.getNoiseLike(img, noisy=True, t=t)

        
        #img_noisy -= img_noisy.min(1, keepdim=True)[0]
        #img_noisy /= img_noisy.max(1, keepdim=True)[0]
        print("IMG NOISy", torch.max(img_noisy), torch.min(img_noisy))
        
        img_noisy = self.make_seed(img_noisy)
        if not eval:
            img_noisy, noise = self.repeatBatch(img_noisy, noise, self.exp.get_from_config('batch_duplication'))
        data_noisy = (id, img_noisy, img_noisy)



        return data_noisy, noise, label

    def get_outputs(self, data, full_img=False, t=0, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        t = torch.tensor((t+1)/self.timesteps).to(self.device)
        print("T", t)
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'), t=t)
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        #return outputs[..., 0:self.output_channels], targets
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        t = torch.randint(0, self.timesteps, (data[1].shape[0],), device=self.exp.get_from_config(tag="device")).long()
        data, noise, label = self.prepare_data(data, t)
        id, img, _ = data
        outputs, _ = self.get_outputs(data, t=t)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}

        #loss = loss_f(outputs, noise)
        print(outputs.dtype, noise.dtype)
        print(outputs.shape, noise.shape)

        loss = F.l1_loss(outputs, noise)
        #loss = F.mse_loss(outputs, noise)

        #loss = torch.mean(torch.sum(torch.square(noise - outputs), dim=(1, 2, 3)) , dim=0)
        #loss = (noise - outputs).square().sum(dim=(1, 2, 3)).mean(dim=0)
        print("LOSS", loss)
        loss_ret[0] = loss

        
        #if len(outputs.shape) == 5:
        #    for m in range(outputs.shape[-1]):
        #        loss_loc = loss_f(outputs[..., m], targets[...])
        #        loss = loss + loss_loc
        #        loss_ret[m] = loss_loc.item()
        #else:
        #    for m in range(outputs.shape[-1]):
        #        if 1 in targets[..., m]:
        #            loss_loc = loss_f(outputs[..., m], targets[..., m])
        #            loss = loss + loss_loc
        #            loss_ret[m] = loss_loc.item()

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss_ret

    def getNoiseLike(self, img, noisy=False, t=0):

        def getNoise():
            rnd = torch.randn_like(img).to(self.device) 
            # Range 0,1
            #rmax, rmin = torch.max(rnd), torch.min(rnd)
            #rnd = ((rnd - rmin) / (rmax - rmin))
            #rnd = rnd*2 -1
            #rnd = rnd*5
            return rnd #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) 
        #noise = torch.rand(img.shape).to(self.device)*2 -1 #orch.tensor.uniform_(noise)
        
        #noise = torch.randn_like(img).to(self.device)
        #noise -= noise.min(0, keepdim=True)[0]
        #noise /= noise.max(0, keepdim=True)[0]
        #noise = noise*2 -1
        
        

        #äprint("NOISE", noise[0].min(), noise[0].max())
        
        if noisy:
            #noise = torch.randn_like(img).to(self.device) #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) #
            noise = getNoise()
            img_noisy = self.q_sample(x_start=img, t=t, noise=noise)
            #img_noisy = torch.clip(img_noisy, -1, 1)
            #noise = img_noisy - img
            img_noisy = img_noisy.to(self.device)
            noise = noise
        else:
            #noise = torch.randn_like(img).to(self.device) #torch.FloatTensor(*img.shape).uniform_(-1, 1).to(self.device) #
            noise = getNoise()
            img_noisy = 0

        return noise.to(self.device), img_noisy

    @torch.no_grad()
    def p_sample(self, output, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        #print(x.shape, betas_t.shape, output.shape, sqrt_one_minus_alphas_cumprod_t.shape)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * output / sqrt_one_minus_alphas_cumprod_t
        )
        # return output
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise, _ = self.getNoiseLike(x) #torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def intermediate_evaluation(self, dataloader, epoch):
        self.test()

    def generateSamples(self, samples=1):
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        #diceLoss = DiceLoss(useSigmoid=useSigmoid)
        self.test(tag="extra", samples=samples, extra=True)

        #return loss_log

    @torch.no_grad()
    def test(self, tag='0', samples=1, extra=False, **kwargs):
        # Generate sample
        size = self.exp.get_from_config('input_size')
        
        #noise = torch.randn_like(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels')))).to(self.device)

        #self.timesteps = 1000
        for s in range(samples):
            noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
            img = self.make_seed(noise)
            for step in reversed(range(self.timesteps)):
                #for i in range(2):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                #print("NOISE HERE", torch.max(img), torch.min(img))
                output, _ = self.get_outputs(img_p, t = step)
                img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img(tag, (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
            #/2 +0.5
        for s in range(samples):
            noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
            img = self.make_seed(noise)
            for step in reversed(range(self.timesteps)):
                for i in range(2):
                    t = torch.full((1,), step, device=self.device, dtype=torch.long)
                    img_p = 0, img, 0
                    #print("NOISE HERE", torch.max(img), torch.min(img))
                    output, _ = self.get_outputs(img_p, t = step)
                    img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                    img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img(tag + '2', (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
            #/2 +0.5
        if False:
            # Extra steps
            for step in reversed(range(int(self.timesteps/2))):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                output, _ = self.get_outputs(img_p, step)
                img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img("extra_steps 50%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep, normalize=True)
            # Extra steps
            for step in reversed(range(int(self.timesteps/2))):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                img_p = 0, img, 0
                output, _ = self.get_outputs(img_p, step)
                img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
            self.exp.write_img("extra_steps 100%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep, normalize=True)
        # For very long runs
        if False:
            for i in range(5):
                for step in reversed(range(int(self.timesteps))):
                    t = torch.full((1,), step, device=self.device, dtype=torch.long)
                    img_p = 0, img, 0
                    output, _ = self.get_outputs(img_p, step)
                    img = self.p_sample(output, img[..., 0:self.exp.get_from_config('input_channels')], t, step)
                    img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img("extra_steps" + str(100 + (i+1)*100) + "%", img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy(), self.exp.currentStep)
        if extra:
            for s in range(samples):
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0]*2, size[1]*2, self.exp.get_from_config('input_channels'))))
                img = self.make_seed(noise)
                for step in reversed(range(self.timesteps)):
                    for i in range(6):
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img("bigger_size", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}

            for s in range(samples):
                noise, _ = self.getNoiseLike(torch.zeros((1, size[0], size[1], self.exp.get_from_config('input_channels'))))
                img = self.make_seed(noise)
                for step in reversed(range(self.timesteps)):
                    for i in range(2):
                        t = torch.full((1,), step, device=self.device, dtype=torch.long)
                        img_p = 0, img, 0
                        #print("NOISE HERE", torch.max(img), torch.min(img))
                        output, _ = self.get_outputs(img_p, t = step)
                        img = self.p_sample(output, img[...,0:self.exp.get_from_config('input_channels')], t, step)
                        img = self.make_seed(img[..., 0:self.exp.get_from_config('input_channels')])
                self.exp.write_img(tag + "doubleSteps", (img[0, ..., 0:self.exp.get_from_config('input_channels')].detach().cpu().numpy()+1)/2, self.exp.currentStep, context={'Image':s}, normalize=True) #/2+0.5 #{'Image':s}
                #/2 +0.5