import torch
import numpy as np
import imageio
from decord import VideoReader
import einops
import torch.nn.functional as F
from transformers import AutoVideoProcessor, AutoModel
import gc
from transformers import Sam2VideoModel, Sam2VideoProcessor
from transformers.video_utils import load_video
from PIL import Image
from diffusers.utils import export_to_video, load_image
import math
from scipy.signal import find_peaks


def tensor2img(tensor):
    # tensor = tensor.float()
    mask_numpy = tensor.cpu().numpy()
    mask_numpy = mask_numpy.astype(np.uint8) * 255

    pil_image = Image.fromarray(mask_numpy)
    return pil_image

def get_video(video_path:str,
              num_frames:int,
              sample_step:int = 1):
    vr = VideoReader(video_path)
    frame_idx = np.arange(0, num_frames, 1)
    video = vr.get_batch(frame_idx).asnumpy()
    return video # T x H x W x C

def vjepa2_video2feature(model,
                         transform,
                         video_path:str, 
                         num_frames:int,
                         device,
                         patch_size = 16,
                         **kwargs):
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video(video_path,num_frames)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W

        # resize to 256*256
        video = F.interpolate(
            video.float(),         # 转换为浮点型
            size=(256, 256),              # 目标尺寸 (H, W)
            mode='bicubic',              # 双线性插值，速度和效果的良好平衡
            align_corners=False           # 推荐设置为 False
        )

        x_hf = transform(video, return_tensors="pt")["pixel_values_videos"].to(device)
        video_embeddings = model.get_vision_features(x_hf)
    return video_embeddings # torch.Size([1, 8192, 1024])

def reward_by_rank(loss: torch.Tensor) -> torch.Tensor:
    """
    reward -> [-1,1]
    """
    # arg
    sorted_indices = torch.argsort(-loss)  # 分数从小到大排列的 index
    ranks = torch.empty_like(sorted_indices)
    
    N = loss.size(0)
    ranks[sorted_indices] = torch.arange(N, device=loss.device)
    
    # rank ∈ [0, N-1] → score ∈ [-1, 1]
    reward = (ranks.float() / (N - 1)) * 2 - 1
    
    return reward

@torch.no_grad()
def reward_vjepa2(gt_video_path:str,
                  sample_videos_path:list[str],
                  modules_config: dict,
                  device,
                  tokens_per_frame=256,
                  is_rank=True,
                  normalize=True):

    model = modules_config["model"]
    transform = modules_config["transform"]
    tokens_per_frame = modules_config["tokens_per_frame"] if "tokens_per_frame" in modules_config.keys() else tokens_per_frame

    # num_frames
    reader = imageio.get_reader(sample_videos_path[0])
    num_frames = reader.count_frames()  
    reader.close()

    # get feature
    gt_feat = vjepa2_video2feature(model = model,
                                   transform = transform,
                                   video_path = gt_video_path,
                                   num_frames = num_frames,
                                   device = device)

    sample_feats = []
    for path in sample_videos_path:
        feat = vjepa2_video2feature(model = model,
                                   transform = transform,
                                   video_path = path,
                                   num_frames = num_frames,
                                   device = device)
        sample_feats.append(feat)
    sample_feats = torch.cat(sample_feats,dim=0) # BSD
    
    # reshape
    gt_feat = einops.rearrange(gt_feat, 'b (p t) d-> b t p d',p=tokens_per_frame) # BTPD
    sample_feats = einops.rearrange(sample_feats, 'b (p t) d-> b t p d',p=tokens_per_frame) # BTPD
    if normalize:
        gt_feat = F.layer_norm(gt_feat, (gt_feat.size(-1),))
        sample_feats = F.layer_norm(sample_feats, (sample_feats.size(-1),))

    mean_dims = tuple(range(1, gt_feat.dim()))
    loss_ft = (gt_feat - sample_feats).abs().mean(dim=mean_dims)*10  # B,
    reward_ft = reward_by_rank(loss_ft) if is_rank else -loss_ft # B,

    # feat graph
    gt_fg = gt_feat.unsqueeze(2) - gt_feat.unsqueeze(1) # BTTPD
    sample_fg = sample_feats.unsqueeze(2) - sample_feats.unsqueeze(1) # BTTPD
    mean_dims = tuple(range(1, gt_fg.dim()))
    loss_fg = (gt_fg - sample_fg).abs().mean(dim=mean_dims)*10  # B,
    reward_fg = reward_by_rank(loss_fg) if is_rank else -loss_fg # B,

    return loss_ft,loss_fg,reward_ft,reward_fg

@torch.no_grad()
def reward_pixel(gt_video_path:str,
                  sample_videos_path:list[str],
                  modules_config: dict,
                  device,
                  tokens_per_frame=256,
                  is_rank=True):

    model = modules_config["model"]
    transform = modules_config["transform"]

    # num_frames
    reader = imageio.get_reader(sample_videos_path[0])
    num_frames = reader.count_frames()  # 大多数情况下有效
    reader.close()

    # get ieo
    gt_video = get_video(gt_video_path,num_frames)  # T x H x W x C
    gt_video = torch.from_numpy(gt_video).permute(0, 3, 1, 2)/255.0  # T x C x H x W
    gt_video = gt_video.unsqueeze(0) # B x T x C x H x W

    sample_videos = []
    for path in sample_videos_path:
        video = get_video(path,num_frames)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)/255.0  # T x C x H x W
        video = video.unsqueeze(0)# B x T x C x H x W
        sample_videos.append(video)
    sample_videos = torch.cat(sample_videos,dim=0) # B x T x C x H x W
    
    # loss_pixel
    mean_dims = tuple(range(1, gt_video.dim()))
    loss_ft = (sample_videos - gt_video).abs().mean(dim=mean_dims).to(device)  # B,
    reward_ft = reward_by_rank(loss_ft).to(device) if is_rank else -loss_ft

    # loss_graph
    gt_fg = gt_video.unsqueeze(2) - gt_video.unsqueeze(1) # BTTC x H x W
    sample_fg = sample_videos.unsqueeze(2) - sample_videos.unsqueeze(1) # BTTC x H x W
    mean_dims = tuple(range(1, gt_fg.dim()))
    loss_fg = (gt_fg - sample_fg).abs().mean(dim=mean_dims).to(device)  # B,
    reward_fg = reward_by_rank(loss_fg).to(device) if is_rank else -loss_fg # B,

    return loss_ft,loss_fg,reward_ft,reward_fg


def detect_collisions(positions, prominence_threshold=2.0, distance_threshold=3):
    """
    collision detection
    :param positions: (N, 2) NumPy 
    :param ball_name
    :param height_threshold
    :param distance_threshold
    :return
    """
    if len(positions) < 3:
        return np.array([]), np.array([])

    
    # velocity
    velocities = np.diff(positions, axis=0)

    
    # accelerations
    accelerations = np.diff(velocities, axis=0)

    
    # acceleration_magnitudes
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

    
    # peaks
    peaks, properties = find_peaks(acceleration_magnitudes, 
                                   prominence=prominence_threshold, 
                                   distance=distance_threshold)
    
    if len(peaks) > 0:
        collision_frames = peaks + 2
    else:
        collision_frames = np.array([])
        
    return acceleration_magnitudes, collision_frames



def get_coords_from_mask(mask):
    img = mask.convert('L')
    img_np = np.array(img)

    brightness_threshold = 127
    is_white = img_np > brightness_threshold
    white_pixel_count = np.sum(is_white)

    # no object
    if white_pixel_count < 10:
        return [-1,-1]

    white_pixels_coords = np.argwhere(is_white)
    center_y, center_x = white_pixels_coords.mean(axis=0)

    return [int(center_x), int(center_y)]


def get_trajectory_from_mask(mask_file_path:str):

    # load frame
    frames_list = []
    reader = imageio.get_reader(mask_file_path)
    for frame in reader:
        frame = Image.fromarray(frame)
        frames_list.append(frame)

    # get traj
    coords_list = []
    for mask in frames_list:
        coord = get_coords_from_mask(mask)
        coords_list.append(coord)
    return coords_list


def export_mask_from_mp4(model,processor,video_path:str,point:list,output_path:str,device):

    video_frames, _ = load_video(video_path)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    # Add click on first frame to select object

    ann_frame_idx = 0
    ann_obj_id = 1
    points = [[[point]]]
    labels = [[[1]]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=points,
        input_labels=labels,
    )

    # Segment the object on the first frame
    outputs = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
    )
    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    # print(f"Segmentation shape: {video_res_masks.shape}")

    # Propagate through the entire video
    video_segments = {}
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
        )[0]
        video_segments[sam2_video_output.frame_idx] = video_res_masks

    # save mask video
    cache = []
    for i in range(len(video_segments)):
        img = tensor2img(video_segments[i][0,0,:]>0)
        cache.append(img)
        
    gc.collect()
    torch.cuda.empty_cache()
    export_to_video(cache, output_path, fps=15, macro_block_size=4)

    return cache

def position_loss(
    gt_traj:list, 
    sample_traj:list, 
    collision_idx, 
    width,
    height,
    normalize=False,
    collision_loss_weight = False,
    ignore_static=False):

    loss_list = []

    # loss_weight 
    loss_weight = [1] * len(gt_traj)
    if collision_loss_weight:
        for idx in collision_idx:
            if idx -1 > 0:
                loss_weight[idx] +=1
            if idx + 1 < len(gt_traj):
                loss_weight[idx] +=1
            loss_weight[idx] +=2
    
    for idx, gt_coord, sample_coord, weight in zip(range(len(gt_traj)), gt_traj, sample_traj, loss_weight):
        # no object in gt frames
        if sample_coord[0]==-1 and sample_coord[1]==-1:
            continue

        # first 5 frames
        if idx <= 4 :
            continue
        
        # static
        if ignore_static and gt_coord[0]==gt_traj[idx-1][0] and gt_coord[1]==gt_traj[idx-1][1]:
            continue
        
        # punish if there is no object in sample frames
        if sample_coord[0]==-1 and sample_coord[1]==-1:
            loss_list.append(200)
            continue

        if normalize:
            sample_coord = (sample_coord[0]/width,sample_coord[1]/height)
            gt_coord = (gt_coord[0]/width,gt_coord[1]/height)
        cur_loss = math.sqrt((sample_coord[0] - gt_coord[0])**2 + (sample_coord[1] - gt_coord[1])**2) * weight
        loss_list.append(cur_loss)

    if len(loss_list) == 0:
        loss_list.append(200)
    loss = sum(loss_list) / len(loss_list)
    return loss




@torch.no_grad()
def reward_position(gt_video_path:str,
                  sample_videos_path:list[str],
                  modules_config: dict,
                  device,
                  width=832,
                  height=480,
                  is_rank=False,
                  mask_file_suffix="_mask",
                  normalize=False,
                  collision_loss_weight=True,
                  ignore_static = True,
                  ):

    # init
    mask_file_suffix = modules_config["mask_file_suffix"] if "mask_file_suffix" in modules_config.keys() else mask_file_suffix
    model = modules_config["model"]
    processor = modules_config["processor"]
    gt_mask_file_path_1 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
    gt_mask_file_path_2 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_2.mp4")

    # get gt trajectory
    gt_traj_1 = get_trajectory_from_mask(gt_mask_file_path_1) # list[Tuple]
    gt_traj_2 = get_trajectory_from_mask(gt_mask_file_path_2)

    # detect collision
    if collision_loss_weight:
        gt_traj_1_np = np.array(gt_traj_1)
        gt_traj_2_np = np.array(gt_traj_2)
        _,collision_idx_1 = detect_collisions(gt_traj_1_np)
        _,collision_idx_2 = detect_collisions(gt_traj_2_np)
    else:
        collision_idx_1 = None
        collision_idx_2 = None
    
    # get point from first frame
    loss_all = []
    for path in sample_videos_path:
        sample_traj_1 = []
        sample_traj_2 = []
        output_path1 = path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
        output_path2 = path.replace(".mp4",f"{mask_file_suffix}_2.mp4")
        frames1 = export_mask_from_mp4(model,processor,path,gt_traj_1[0],output_path1,device)
        frames2 = export_mask_from_mp4(model,processor,path,gt_traj_2[0],output_path2,device)
        sample_traj_1 = (get_trajectory_from_mask(output_path1))
        sample_traj_2 = (get_trajectory_from_mask(output_path2))
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight,ignore_static)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight,ignore_static)
        loss_all.append(0.5*loss1 + 0.5*loss2)
    loss_tensor = torch.tensor(loss_all,device=device) 

    if is_rank:
        reward = reward_by_rank(loss_tensor)
    else:
        reward = -loss_tensor
    
    return loss_tensor,loss_tensor,reward,reward


@torch.no_grad()
def reward_position_v2v_metric(gt_video_path:str,
                  sample_videos_path:list[str],
                  modules_config: dict,
                  device,
                  width=832,
                  height=480,
                  is_rank=False,
                  mask_file_suffix="_mask",
                  normalize=False,
                  collision_loss_weight=True,
                  ignore_static = True,
                  ):

    # init
    mask_file_suffix = modules_config["mask_file_suffix"] if "mask_file_suffix" in modules_config.keys() else mask_file_suffix
    model = modules_config["model"]
    processor = modules_config["processor"]
    gt_mask_file_path_1 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
    gt_mask_file_path_2 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_2.mp4")

    # get gt trajectory
    gt_traj_1 = get_trajectory_from_mask(gt_mask_file_path_1) # list[Tuple]
    gt_traj_2 = get_trajectory_from_mask(gt_mask_file_path_2)

    # detect collision
    gt_traj_1_np = np.array(gt_traj_1)
    gt_traj_2_np = np.array(gt_traj_2)
    _,collision_idx_1 = detect_collisions(gt_traj_1_np)
    _,collision_idx_2 = detect_collisions(gt_traj_2_np)
    
    # get point from first frame
    loss_dict = {'type1':[],'type2':[],'type3':[],'type4':[]}
    for path in sample_videos_path:
        sample_traj_1 = []
        sample_traj_2 = []
        output_path1 = path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
        output_path2 = path.replace(".mp4",f"{mask_file_suffix}_2.mp4")
        frames1 = export_mask_from_mp4(model,processor,path,gt_traj_1[0],output_path1,device)
        frames2 = export_mask_from_mp4(model,processor,path,gt_traj_2[0],output_path2,device)
        sample_traj_1 = (get_trajectory_from_mask(output_path1))
        sample_traj_2 = (get_trajectory_from_mask(output_path2))

        # type 1 
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight=False,ignore_static=False)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight=False,ignore_static=False)
        loss_dict['type1'].append(0.5*loss1 + 0.5*loss2)

        # type 2 
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight=False,ignore_static=True)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight=False,ignore_static=True)
        loss_dict['type2'].append(0.5*loss1 + 0.5*loss2)        

        # type 3
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight=True,ignore_static=False)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight=True,ignore_static=False)
        loss_dict['type3'].append(0.5*loss1 + 0.5*loss2) 

        # type 4 
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight=True,ignore_static=True)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight=True,ignore_static=True)
        loss_dict['type4'].append(0.5*loss1 + 0.5*loss2) 


    for k,v in loss_dict.items():
        loss_dict[k] = sum(v) / len(v)
    
    return loss_dict



@torch.no_grad()
def calculate_iou_trajoffset(gt_video_path:str,
                  sample_videos_path:list[str],
                  modules_config: dict,
                  device,
                  width=832,
                  height=480,
                  is_rank=False,
                  mask_file_suffix="_mask",
                  normalize=False,
                  collision_loss_weight=True,
                  ignore_static = True,
                  ):

    # init
    mask_file_suffix = modules_config["mask_file_suffix"] if "mask_file_suffix" in modules_config.keys() else mask_file_suffix
    model = modules_config["model"]
    processor = modules_config["processor"]
    gt_mask_file_path_1 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
    gt_mask_file_path_2 = gt_video_path.replace(".mp4",f"{mask_file_suffix}_2.mp4")

    # get gt trajectory
    gt_traj_1 = get_trajectory_from_mask(gt_mask_file_path_1) # list[Tuple]
    gt_traj_2 = get_trajectory_from_mask(gt_mask_file_path_2)

    # detect collision
    if collision_loss_weight:
        gt_traj_1_np = np.array(gt_traj_1)
        gt_traj_2_np = np.array(gt_traj_2)
        _,collision_idx_1 = detect_collisions(gt_traj_1_np)
        _,collision_idx_2 = detect_collisions(gt_traj_2_np)
    else:
        collision_idx_1 = None
        collision_idx_2 = None
    
    # get point from first frame
    loss_all = []
    for path in sample_videos_path:
        sample_traj_1 = []
        sample_traj_2 = []
        output_path1 = path.replace(".mp4",f"{mask_file_suffix}_1.mp4")
        output_path2 = path.replace(".mp4",f"{mask_file_suffix}_2.mp4")
        frames1 = export_mask_from_mp4(model,processor,path,gt_traj_1[0],output_path1,device)
        frames2 = export_mask_from_mp4(model,processor,path,gt_traj_2[0],output_path2,device)
        sample_traj_1 = (get_trajectory_from_mask(output_path1))
        sample_traj_2 = (get_trajectory_from_mask(output_path2))
        loss1 = position_loss(gt_traj_1,sample_traj_1,collision_idx_1,width,height,normalize,collision_loss_weight,ignore_static)
        loss2 = position_loss(gt_traj_2,sample_traj_2,collision_idx_2,width,height,normalize,collision_loss_weight,ignore_static)
        loss_all.append(0.5*loss1 + 0.5*loss2)
    loss_tensor = torch.tensor(loss_all,device=device) 

    if is_rank:
        reward = reward_by_rank(loss_tensor)
    else:
        reward = -loss_tensor
    
    return loss_tensor,loss_tensor,reward,reward

