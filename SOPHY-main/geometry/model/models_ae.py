import torch
from torch import nn
from torch_cluster import fps

from model.common_ae import (
    cache_fn, exists,
    PreNorm,
    FeedForward,
    MLPHead,
    Attention,
    PointEmbed,
    DiagonalGaussianDistribution,
)

class KLAutoEncoder(nn.Module):
    """
    Input features:
        Position, Color, Material
    Latent features:
        Occupancy, Color, Material
    Output features:
        Occupancy, Color, Material
    """
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        num_mat_classes = None,
        mat_decoder_ff = False,
        mat_layers = 0,
        rgb_embed_dim = 0,
        part_embed_dim = 0,
        mat_embed_dim = 0,
        num_part_classes = 0,
        part_loss = False,
        mat_start_depth = None,
        requires_e = False,     # Young's modulus
        requires_nu = False,    # Poisson ratio
        requires_sigma = False, # Yield stress
        requires_phi = False,   # Friction angle
        requires_rho = False,   # Density
        num_mmid = 0,           # nums of Material Models
        rgb_layers = 0,
        rgb_decoder_ff = False,
        **kwargs
    ):
        super().__init__()

        self.depth = depth
        self.latent_dim = latent_dim

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.num_mat_layers = mat_layers
        self.num_rgb_layers = rgb_layers

        self.requires_e = requires_e
        self.requires_nu = requires_nu
        self.requires_sigma = requires_sigma
        self.requires_phi = requires_phi
        self.requires_rho = requires_rho
        # input dimensions for material features (e, nu, sigma, phi, rho, mat model id)
        self.dim_mat_input = 1 + requires_e + requires_nu + requires_sigma + requires_phi + requires_rho
        self.num_mmid = num_mmid

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        # various embeddings
        self.point_embed = PointEmbed(
            dim=dim, rgb_embed_dim=rgb_embed_dim,
            part_embed_dim=part_embed_dim, num_part_classes=num_part_classes,
            dim_mat_input=self.dim_mat_input, mat_embed_dim=mat_embed_dim, num_mmid=num_mmid
        )

        self.use_part_loss = part_loss

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.num_mat_classes = num_mat_classes
        self.num_part_classes = num_part_classes
        self.mat_start_depth = mat_start_depth
        if mat_layers > 0:
            self.mat_layers = nn.ModuleList([])
            for _ in range(mat_layers):
                self.mat_layers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))
            self.mat_decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
            self.mat_decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if mat_decoder_ff else None

            self.mat_proj = nn.Linear(latent_dim, dim)

        if num_part_classes is not None and num_part_classes > 0:
            self.to_part_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, num_part_classes)
            )

        if num_mat_classes is not None and num_mat_classes > 0:
            self.to_mat_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, num_mat_classes)
            )

        if requires_e:  # NOTE: linear activation
            self.to_E_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, 1)
            )

        if requires_nu:
            self.to_nu_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, 1),
                nn.Sigmoid()
            )

        if requires_sigma:  # NOTE: linear activation
            self.to_sigma_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, 1)
            )

        if requires_phi:
            self.to_phi_outputs = nn.Sequential(
                PreNorm(queries_dim, MLPHead(queries_dim, queries_dim, drop_rate=0.1)),
                nn.Linear(queries_dim, 1),
                nn.Sigmoid()
            )

        if requires_rho:
            self.to_rho_outputs = nn.Sequential(
                PreNorm(queries_dim, MLPHead(queries_dim, queries_dim, drop_rate=0.1)),
                nn.Linear(queries_dim, 1),
                nn.ReLU(inplace=True)
            )

        if num_mmid > 0:
            self.to_mmid_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, num_mmid),
            )

        if rgb_layers > 0:
            self.rgb_layers = nn.ModuleList([])
            for _ in range(rgb_layers):
                self.rgb_layers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))
            self.rgb_decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
            self.rgb_decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if rgb_decoder_ff else None
            self.to_rgb_outputs = nn.Sequential(
                PreNorm(queries_dim, FeedForward(queries_dim, mult = 1)),
                nn.Linear(queries_dim, 3),   # NOTE: Use linear activation
            )

            self.rgb_proj = nn.Linear(latent_dim, dim)

        self.proj = nn.Linear(latent_dim, dim)

        latent_dim_essence = latent_dim
        if mat_layers > 0:
            latent_dim_essence += latent_dim
        if rgb_layers > 0:
            latent_dim_essence += latent_dim

        self.mean_fc = nn.Linear(dim, latent_dim_essence)
        self.logvar_fc = nn.Linear(dim, latent_dim_essence)

    def encode(self, pc, **kwargs):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        rgb = kwargs.get('rgb')
        part = kwargs.get('part')
        mat = kwargs.get('mat')
        if self.point_embed.rgb_embed_dim > 0:
            assert rgb.shape == (B, N, D), rgb.shape
        if self.point_embed.part_embed_dim > 0:
            assert part.shape == (B, N), part.shape
        if self.point_embed.mat_embed_dim > 0:
            assert mat.shape == (B, N, self.dim_mat_input), mat.shape

        ###### fps
        flattened = pc.view(B*N, D)
        flattened_rgb = rgb.view(B*N, D) if exists(rgb) else None
        flattened_part = part.view(B*N, 1) if exists(part) else None
        flattened_mat = mat.view(B*N, self.dim_mat_input) if exists(mat) else None

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        if exists(rgb):
            sampled_rgb = flattened_rgb[idx]
            sampled_rgb = sampled_rgb.view(B, -1, 3)
        else:
            sampled_rgb = None
        if exists(part):
            sampled_part = flattened_part[idx]
            sampled_part = sampled_part.view(B, -1)
        else:
            sampled_part = None
        if exists(mat):
            sampled_mat = flattened_mat[idx]
            sampled_mat = sampled_mat.view(B, -1, self.dim_mat_input)
        else:
            sampled_mat = None
        ######

        # encode input features
        sampled_pc_embeddings = self.point_embed(sampled_pc, rgb=sampled_rgb, part=sampled_part, mat=sampled_mat)
        pc_embeddings = self.point_embed(pc, rgb=rgb, part=part, mat=mat)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        out = {'kl': kl, 'latents': x}

        return out

    def decode_rgb(self, x, queries):
        rgb_x = x[..., self.latent_dim * 2:]
        rgb_x = self.rgb_proj(rgb_x)
        occ_x = x[..., :self.latent_dim]
        occ_x = self.proj(occ_x)

        if self.mat_start_depth is None:
            mat_start_depth = len(self.layers) // 2
        else:
            mat_start_depth = self.mat_start_depth + 1

        for idx in range(mat_start_depth):  # 0, ..., mat_start_depth - 1
            self_attn, self_ff = self.layers[idx]
            self_attn_rgb, self_ff_rgb = self.rgb_layers[idx]

            if idx % 2 == 0:
                rgb_x = self_attn_rgb(rgb_x) + rgb_x
            else:
                rgb_x = self_attn_rgb(rgb_x, context = occ_x) + rgb_x
            rgb_x = self_ff_rgb(rgb_x) + rgb_x

            occ_x = self_attn(occ_x) + occ_x
            occ_x = self_ff(occ_x) + occ_x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)

        # additional layers for rgb prediction
        for idx in range(mat_start_depth, self.num_rgb_layers):
            self_attn_rgb, self_ff_rgb = self.rgb_layers[idx]
            rgb_x = self_attn_rgb(rgb_x) + rgb_x
            rgb_x = self_ff_rgb(rgb_x) + rgb_x
        # cross attend from decoder queries to latents
        rgb_latents = self.rgb_decoder_cross_attn(queries_embeddings, context = rgb_x)
        if exists(self.rgb_decoder_ff):
            rgb_latents = self.rgb_decoder_ff(rgb_latents)
        out = self.to_rgb_outputs(rgb_latents)

        return out

    def decode(self, x, queries, **kwargs):

        # NOTE: Currently accept two settings:
        # 1. decode shape + rgb + material
        # 2. decode shape only (when num_mat_layers = 0 and num_rgb_layers = 0)

        # split latents into shape, rgb, material subcodes
        if self.num_mat_layers > 0 and self.num_rgb_layers > 0:
            mat_x = x[..., self.latent_dim:self.latent_dim * 2]
            mat_x = self.mat_proj(mat_x)
            rgb_x = x[..., self.latent_dim * 2:]
            rgb_x = self.rgb_proj(rgb_x)
            occ_x = x[..., :self.latent_dim]
            occ_x = self.proj(occ_x)
        else:
            occ_x = self.proj(x)

        if self.num_mat_layers > 0 and self.num_rgb_layers > 0:

            # forward interaction block (shape, rgb, material)

            if self.mat_start_depth is None:
                mat_start_depth = len(self.layers) // 2
            else:
                mat_start_depth = self.mat_start_depth + 1

            for idx in range(mat_start_depth):  # 0, ..., mat_start_depth - 1
                self_attn, self_ff = self.layers[idx]
                self_attn_rgb, self_ff_rgb = self.rgb_layers[idx]
                self_attn_mat, self_ff_mat = self.mat_layers[idx]

                if idx % 2 == 0:
                    mat_x = self_attn_mat(mat_x) + mat_x
                else:
                    context_mat_x = torch.cat([occ_x, rgb_x], dim=1)
                    mat_x = self_attn_mat(mat_x, context = context_mat_x) + mat_x
                mat_x = self_ff_mat(mat_x) + mat_x

                if idx % 2 == 0:
                    rgb_x = self_attn_rgb(rgb_x) + rgb_x
                else:
                    rgb_x = self_attn_rgb(rgb_x, context = occ_x) + rgb_x
                rgb_x = self_ff_rgb(rgb_x) + rgb_x

                occ_x = self_attn(occ_x) + occ_x
                occ_x = self_ff(occ_x) + occ_x

            # additional layers for shape prediction
            for idx in range(mat_start_depth, len(self.layers)):
                self_attn, self_ff = self.layers[idx]
                occ_x = self_attn(occ_x) + occ_x
                occ_x = self_ff(occ_x) + occ_x

        else:

            for self_attn, self_ff in self.layers:
                occ_x = self_attn(occ_x) + occ_x
                occ_x = self_ff(occ_x) + occ_x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = occ_x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        occ_logits = self.to_outputs(latents)

        out = {'logits': occ_logits}

        if self.num_rgb_layers > 0 and not kwargs.get('wo_rgb', False):

            # additional layers for rgb prediction
            for idx in range(mat_start_depth, self.num_rgb_layers):
                self_attn_rgb, self_ff_rgb = self.rgb_layers[idx]
                rgb_x = self_attn_rgb(rgb_x) + rgb_x
                rgb_x = self_ff_rgb(rgb_x) + rgb_x

            if kwargs.get('queries_for_rgb') is not None:
                # so we can query surface points for rgb prediction during training
                queries_for_rgb_embeddings = self.point_embed(kwargs['queries_for_rgb'])
            else:
                queries_for_rgb_embeddings = queries_embeddings
            # cross attend from decoder queries to latents
            rgb_latents = self.rgb_decoder_cross_attn(queries_for_rgb_embeddings, context = rgb_x)
            if exists(self.rgb_decoder_ff):
                rgb_latents = self.rgb_decoder_ff(rgb_latents)
            out['rgb_logits'] = self.to_rgb_outputs(rgb_latents)

        if self.num_mat_layers > 0:

            # additional layers for material prediction
            for idx in range(mat_start_depth, self.num_mat_layers):
                self_attn_mat, self_ff_mat = self.mat_layers[idx]
                mat_x = self_attn_mat(mat_x) + mat_x
                mat_x = self_ff_mat(mat_x) + mat_x

            # cross attend from decoder queries to latents
            mat_latents = self.mat_decoder_cross_attn(queries_embeddings, context = mat_x)
            if self.num_part_classes is not None and self.num_part_classes > 0:
                out['part_logits'] = self.to_part_outputs(mat_latents)

            # optional decoder feedforward
            if exists(self.mat_decoder_ff):
                mat_latents = self.mat_decoder_ff(mat_latents)
            if self.training and self.use_part_loss:
                out['mat_latents'] = mat_latents.clone()

            if self.num_mat_classes is not None and self.num_mat_classes > 0:
                mat_logits = self.to_mat_outputs(mat_latents)
                out['mat_logits'] = mat_logits
            if self.requires_e:
                out['mat_Es'] = self.to_E_outputs(mat_latents)
            if self.requires_nu:
                out['mat_nus'] = self.to_nu_outputs(mat_latents)
            if self.requires_sigma:
                out['mat_sigmas'] = self.to_sigma_outputs(mat_latents)
            if self.requires_phi:
                out['mat_phis'] = self.to_phi_outputs(mat_latents) * 0.5 * torch.pi
            if self.requires_rho:
                out['mat_rhos'] = self.to_rho_outputs(mat_latents)
            if self.num_mmid > 0:
                out['mat_mmid_logits'] = self.to_mmid_outputs(mat_latents)
 
        return out

    def forward(self, pc, queries, **kwargs):
        enc_out = self.encode(
            pc, rgb=kwargs.get('pc_rgb'), part=kwargs.get('pc_part'), mat=kwargs.get('pc_mat'),
        )
        kl, x = enc_out['kl'], enc_out['latents']

        dec_kwargs = dict()
        if kwargs.get('queries_for_rgb') is not None:
            dec_kwargs['queries_for_rgb'] = kwargs['queries_for_rgb']
        dec_out = self.decode(x, queries, **dec_kwargs)

        out = {'logits': dec_out['logits'].squeeze(-1), 'kl': kl}

        if dec_out.get('mat_logits') is not None:
            out['mat_logits'] = dec_out['mat_logits']
        if dec_out.get('part_logits') is not None:
            out['part_logits'] = dec_out['part_logits']
        if dec_out.get('mat_Es') is not None:
            out['mat_Es'] = dec_out['mat_Es']
        if dec_out.get('mat_nus') is not None:
            out['mat_nus'] = dec_out['mat_nus']
        if dec_out.get('mat_sigmas') is not None:
            out['mat_sigmas'] = dec_out['mat_sigmas']
        if dec_out.get('mat_phis') is not None:
            out['mat_phis'] = dec_out['mat_phis']
        if dec_out.get('mat_rhos') is not None:
            out['mat_rhos'] = dec_out['mat_rhos']
        if dec_out.get('mat_mmid_logits') is not None:
            out['mat_mmid_logits'] = dec_out['mat_mmid_logits']
        if dec_out.get('rgb_logits') is not None:
            out['rgb_logits'] = dec_out['rgb_logits']
        if dec_out.get('mat_latents') is not None:
            if self.training and self.use_part_loss:
                out['mat_latents'] = dec_out['mat_latents']

        return out

def create_autoencoder(
    dim=512,
    M=512,
    latent_dim=64,
    N=2048,
    decoder_ff=False,
    num_mat_classes=None,
    mat_decoder_ff=False,
    mat_layers = 0,
    rgb_embed_dim = 0,
    part_embed_dim = 0,
    mat_embed_dim = 0,
    num_part_classes = 0,
    part_loss = False,
    mat_start_depth=None,
    requires_e=False,
    requires_nu=False,
    requires_sigma=False,
    requires_phi=False,
    requires_rho=False,
    num_mmid=0,
    rgb_layers=0,
    rgb_decoder_ff=False,
):
    model = KLAutoEncoder(
        depth=24,
        dim=dim,
        queries_dim=dim,
        output_dim = 1,
        num_inputs = N,
        num_latents = M,
        latent_dim = latent_dim,
        heads = 8,
        dim_head = 64,
        decoder_ff = decoder_ff,
        num_mat_classes = num_mat_classes,
        mat_decoder_ff = mat_decoder_ff,
        mat_layers=mat_layers,
        rgb_embed_dim = rgb_embed_dim,
        part_embed_dim = part_embed_dim,
        mat_embed_dim = mat_embed_dim,
        num_part_classes = num_part_classes,
        part_loss = part_loss,
        mat_start_depth=mat_start_depth,
        requires_e=requires_e,
        requires_nu=requires_nu,
        requires_sigma=requires_sigma,
        requires_phi=requires_phi,
        requires_rho=requires_rho,
        num_mmid=num_mmid,
        rgb_layers=rgb_layers,
        rgb_decoder_ff=rgb_decoder_ff,
    )
    return model

def kl_d512_m512_l16_mat(
    N=2048,
    decoder_ff=False,
    num_mat_classes=None,
    mat_decoder_ff=False,
    mat_layers=0,
    rgb_embed_dim=0,
    part_embed_dim=0,
    mat_embed_dim=0,
    num_part_classes=0,
    part_loss=False,
    mat_start_depth=None,
    requires_e=False,
    requires_nu=False,
    requires_sigma=False,
    requires_phi=False,
    requires_rho=False,
    num_mmid=0,
    rgb_layers=0,
    rgb_decoder_ff=False,
):
    return create_autoencoder(
        dim=512, M=512, latent_dim=16, N=N,
        decoder_ff=decoder_ff,
        num_mat_classes=num_mat_classes,
        mat_decoder_ff=mat_decoder_ff,
        mat_layers=mat_layers,
        rgb_embed_dim=rgb_embed_dim,
        part_embed_dim=part_embed_dim,
        mat_embed_dim=mat_embed_dim,
        num_part_classes=num_part_classes,
        part_loss=part_loss,
        mat_start_depth=mat_start_depth,
        requires_e=requires_e,
        requires_nu=requires_nu,
        requires_sigma=requires_sigma,
        requires_phi=requires_phi,
        requires_rho=requires_rho,
        num_mmid=num_mmid,
        rgb_layers=rgb_layers,
        rgb_decoder_ff=rgb_decoder_ff,
    )

def kl_d512_m512_l8_mat(
    N=2048,
    decoder_ff=False,
    num_mat_classes=None,
    mat_decoder_ff=False,
    mat_layers=0,
    rgb_embed_dim=0,
    part_embed_dim=0,
    mat_embed_dim=0,
    num_part_classes=0,
    part_loss=False,
    mat_start_depth=None,
    requires_e=False,
    requires_nu=False,
    requires_sigma=False,
    requires_phi=False,
    requires_rho=False,
    num_mmid=0,
    rgb_layers=0,
    rgb_decoder_ff=False,
):
    return create_autoencoder(
        dim=512, M=512, latent_dim=8, N=N,
        decoder_ff=decoder_ff,
        num_mat_classes=num_mat_classes,
        mat_decoder_ff=mat_decoder_ff,
        mat_layers=mat_layers,
        rgb_embed_dim=rgb_embed_dim,
        part_embed_dim=part_embed_dim,
        mat_embed_dim=mat_embed_dim,
        num_part_classes=num_part_classes,
        part_loss=part_loss,
        mat_start_depth=mat_start_depth,
        requires_e=requires_e,
        requires_nu=requires_nu,
        requires_sigma=requires_sigma,
        requires_phi=requires_phi,
        requires_rho=requires_rho,
        num_mmid=num_mmid,
        rgb_layers=rgb_layers,
        rgb_decoder_ff=rgb_decoder_ff,
    )
