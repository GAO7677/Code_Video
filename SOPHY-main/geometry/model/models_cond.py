import torch
import torch.nn as nn

from model.common_cond import (
    edm_sampler,
    EDMLoss,
    LatentArrayTransformer,
    StackedRandomGenerator
)

class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        n_latents = 512,
        channels = 8, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        n_heads = 8,
        d_head = 64,
        depth = 12,
        conditional_signal = None,
        conditional_arg = None,
        # depth = 6,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.conditional_signal = conditional_signal

        self.model = LatentArrayTransformer(
            in_channels=channels, t_channels=256, n_heads=n_heads, d_head=d_head, depth=depth,
            context_dim=conditional_arg if conditional_signal is not None and conditional_signal != 'category' else None,
        )

        if conditional_signal == 'category':
            self.category_emb = nn.Embedding(conditional_arg, n_heads * d_head)
        else:
            self.category_emb = nn.Identity()

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, x, sigma, cond_signals=None, force_fp32=False, **model_kwargs):

        if self.conditional_signal is None:
            cond_emb = None
        elif self.conditional_signal == 'category':
            if cond_signals.dtype == torch.float32:
                cond_emb = cond_signals
            else:
                cond_emb = self.category_emb(cond_signals).unsqueeze(1)
        elif self.conditional_signal in ['text', 'image']:
            cond_emb = cond_signals
        else:
            raise ValueError(f'Unknown conditional signal: {self.conditional_signal}')

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    @torch.no_grad()
    def sample(self, cond, batch_seeds=None):
        # print(batch_seeds)
        if cond is not None:
            batch_size, device = cond.shape[0], cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        # batch_size, device = *cond.shape, cond.device
        # batch_seeds = torch.arange(batch_size)

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)

        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like)

def kl_d512_m512_l8_d24_edm(conditional_signal,  conditional_arg):
    model = EDMPrecond(conditional_signal=conditional_signal, conditional_arg=conditional_arg, n_latents=512, channels=8, depth=24)
    return model

def kl_d512_m512_l16_d24_edm(conditional_signal,  conditional_arg):
    model = EDMPrecond(conditional_signal=conditional_signal, conditional_arg=conditional_arg, n_latents=512, channels=16, depth=24)
    return model

def kl_d512_m512_l24_d24_edm(conditional_signal,  conditional_arg):
    model = EDMPrecond(conditional_signal=conditional_signal, conditional_arg=conditional_arg, n_latents=512, channels=24, depth=24)
    return model