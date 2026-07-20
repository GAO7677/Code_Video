from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    StridedRandomSliceSequence,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    YTVIS,
    ClPadToMax1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    Adam,
    ClipGradNorm,
    MSELoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    Callback,
    AverageLog,
)
from object_centric_bench.model import (
    RandSFQ2,
    Sequential,
    Identity,
    MLP,
    NormalShared,
    SlotAttention,
    RSFQTransit,
    MarkovRarDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from object_centric_bench.model.dinov3_backbone import DINO3ViT
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

# Central experiment controls. Keep derived module dimensions below tied to
# these values so slot-width ablations do not require model-code changes.
variant_name = "dinov3_vitl16_lvd1689m_slot512"
num_slots = 7
slot_dim = 512
backbone_feature_dim = 1024
resolut0 = [256, 256]
resolut1 = [16, 16]
train_clip_frames = 5
dataset_balance_frames = 30
train_crop_scale = [0.75, 1.0]
train_flip_prob = 0.5

encoder_project_dropout = 0.0
slot_num_iter = 3
slot_ffn_expansion = 4
slot_dropout = 0.0
slot_trunc_bp = None

transition_dt = 5
transition_num_heads = 8
transition_ffn_expansion = 4
transition_dropout = 0.5
transition_norm_first = False
transition_bias = False

decoder_dt = 1
decoder_dynamic_ratio = 0.25
decoder_num_heads = 4
decoder_ffn_expansion = 4
decoder_dropout = 0.0
decoder_num_layers = 4
decoder_norm_first = True
decoder_bias = False

# Compatibility names consumed by the official xSSC code and downstream loader.
max_num = num_slots
emb_dim = slot_dim
vfm_dim = backbone_feature_dim

total_step = 50000  # 100000 better
max_step = total_step
gpu_ids = [0, 1, 2, 3]
expected_world_size = len(gpu_ids)
amp_dtype = "bfloat16"
distributed_backend = "nccl"
distributed_timeout_minutes = 60
train_sampler_drop_last = False
train_loader_drop_last = False
cudnn_benchmark = False
cudnn_deterministic = True
use_deterministic_algorithms = True
num_validation_runs = 40
warmup_fraction = 0.05
final_lr_ratio = 1e-3
checkpoint_interval = 1000
val_interval = total_step // num_validation_runs
batch_size_t = 96  # per GPU; 4-GPU DDP global batch = 384
batch_size_v = 1
num_work = 4
lr = 2e-4 / 4  # scale with batch_size
gradient_clip_norm = 0.05

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=train_crop_scale),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=train_flip_prob),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=YTVIS,
    data_file="ytvis_hq/train.lmdb",
    extra_keys=["segment"],
    transform0=dict(
        type=StridedRandomSliceSequence,
        keys=["video", "segment"],
        size=train_clip_frames,
    ),
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
    ts=dataset_balance_frames,
)
dataset_v = dict(
    type=YTVIS,
    data_file="ytvis_hq/val.lmdb",
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

model = dict(
    type=RandSFQ2,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(
                type=DINO3ViT,
                model_name="dinov3_vitl16",
                in_size=resolut0[0],
                rearrange=True,
                norm_out=False,
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP,
        in_dim=vfm_dim,
        dims=[vfm_dim, vfm_dim],
        ln="pre",
        dropout=encoder_project_dropout,
    ),
    initializ=dict(type=NormalShared, num=max_num, dim=emb_dim),  # >NormalSeparat
    aggregat=dict(
        type=SlotAttention,
        num_iter=slot_num_iter,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * slot_ffn_expansion,
        dropout=slot_dropout,
        kv_dim=vfm_dim,
        trunc_bp=slot_trunc_bp,
    ),
    transit=dict(
        type=RSFQTransit,
        dt=transition_dt,
        ci=vfm_dim,
        c=emb_dim,
        nhead=transition_num_heads,
        expanz=transition_ffn_expansion,
        pdo=transition_dropout,
        norm_first=transition_norm_first,
        bias=transition_bias,
    ),
    decode=dict(
        type=MarkovRarDecoder,
        dt=decoder_dt,
        rd=decoder_dynamic_ratio,
        emb_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(  # fc>fc+ln
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(  # fc+ln>fc
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=decoder_num_heads,
                dim_feedforward=vfm_dim * decoder_ffn_expansion,
                dropout=decoder_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=decoder_norm_first,
                bias=decoder_bias,
            ),
            num_layers=decoder_num_layers,
        ),
        readout=dict(type=Identity),
    ),
)
model_imap = dict(input="batch.video")
model_omap = ["feature", "slotz", "attenta", "recon", "attentd"]
ckpt_map = []  # target<-source
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gclip = dict(type=ClipGradNorm, max_norm=gradient_clip_norm)

loss_fn_t = loss_fn_v = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=int(total_step * warmup_fraction),
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr * final_lr_ratio,
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attentd"]],  # (b,t,s,h,w) -> (b,t,h,w,s)
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
]
