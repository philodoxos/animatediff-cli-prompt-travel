import sys
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch

sys.path.append('/workspace/latent-diffusion')
sys.path.append('/workspace/latent-diffusion/ldm')

from ldm.modules.diffusionmodules.util import (extract_into_tensor,
                                               make_beta_schedule)


class FakeDDPM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.num_timesteps = 1000
        self.beta_start = 0.00085
        self.beta_end=0.0120
        self.cosine_s=8e-3

        betas = make_beta_schedule("linear", self.num_timesteps, self.beta_start, self.beta_end, self.cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))


    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract_into_tensor(self.sqrt_recip_alphas_cumprod.to(x_t), t, x_t.shape) * x_t - x0) /
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod.to(x_t), t, x_t.shape)
        )

fake_ddpm = FakeDDPM()

def colorfix_blur(x, k):
    y = torch.nn.functional.pad(x, (k, k, k, k), mode='replicate')
    y = torch.nn.functional.avg_pool2d(y, (k*2+1, k*2+1), stride=(1, 1))
    return y

def colorfix_noise(latent_xt, colorfix_latent, colorfix_weight, colorfix_variation, cur_noise, timesteps):
    xt = latent_xt[:, :4, :, :]
    x0_origin = colorfix_latent
    k = colorfix_variation
    w = max(0.0, min(1.0, float(colorfix_weight)))
    t = torch.round(timesteps.float()).long()
    h = cur_noise

    x0_prd = fake_ddpm.predict_start_from_noise(xt, t, h)
    x0 = x0_prd - colorfix_blur(x0_prd, k) + colorfix_blur(x0_origin, k)

    eps_prd = fake_ddpm.predict_noise_from_start(xt, t, x0)

    h = eps_prd * w + h * (1 - w)

    return h