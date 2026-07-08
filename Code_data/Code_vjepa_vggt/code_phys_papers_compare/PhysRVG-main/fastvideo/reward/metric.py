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
import torchvision
import os

def tensor2img(tensor):
    # tensor = tensor.float()
    mask_numpy = tensor.cpu().numpy()
    mask_numpy = mask_numpy.astype(np.uint8) * 255

    pil_image = Image.fromarray(mask_numpy)
    return pil_image

def get_coords_from_mask(mask):
    img = mask.convert('L')
    img_np = np.array(img)

    brightness_threshold = 127
    is_white = img_np > brightness_threshold
    white_pixel_count = np.sum(is_white)

    # 找不到主体
    if white_pixel_count < 10:
        return [-1,-1],is_white

    white_pixels_coords = np.argwhere(is_white)
    center_y, center_x = white_pixels_coords.mean(axis=0)

    return [int(center_x), int(center_y)],is_white


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
        coord,_ = get_coords_from_mask(mask)
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
    print(f"Segmentation shape: {video_res_masks.shape}")

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

    # loss_weight 【是否是撞击瞬间】决定
    loss_weight = [1] * len(gt_traj)
    if collision_loss_weight:
        for idx in collision_idx:
            if idx -1 > 0:
                loss_weight[idx] +=1
            if idx + 1 < len(gt_traj):
                loss_weight[idx] +=1
            loss_weight[idx] +=2
    
    for idx, gt_coord, sample_coord, weight in zip(range(len(gt_traj)), gt_traj, sample_traj, loss_weight):
        print(f"-------- {idx} ----------")
        # gt都检测不到主体，跳过
        if sample_coord[0]==-1 and sample_coord[1]==-1:
            print("没主体")
            continue

        # 前五帧，跳过
        if idx <= 4 :
            print("前5帧")
            continue
        
        # 物体是静止的，跳过
        if ignore_static and gt_coord[0]==gt_traj[idx-1][0] and gt_coord[1]==gt_traj[idx-1][1]:
            print("静止的")
            continue
        
        # 如果sample检测不到主体，惩罚
        if sample_coord[0]==-1 and sample_coord[1]==-1:
            print("惩罚")
            loss_list.append(200)
            continue
        print("正常")
        if normalize:
            sample_coord = (sample_coord[0]/width,sample_coord[1]/height)
            gt_coord = (gt_coord[0]/width,gt_coord[1]/height)
        cur_loss = math.sqrt((sample_coord[0] - gt_coord[0])**2 + (sample_coord[1] - gt_coord[1])**2) * weight
        print(f"loss{cur_loss}")
        loss_list.append(cur_loss)

    if len(loss_list) == 0:
        loss_list.append(50)
    loss = sum(loss_list) / len(loss_list)
    return loss


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image


def load_frames(video_path,num_frames=49,step=2):
    reader = imageio.get_reader(video_path)
    frames = []
    for i,f in enumerate(reader):
      if len(frames) >= num_frames:
        break

      if i%step == 0:
        frame = Image.fromarray(f)
        frame = crop_and_resize(frame,480,832)
        frames.append(frame)
    reader.close()
    return frames

def compute_iou(mask_a, mask_b):
    """计算单帧 IoU，mask_a/mask_b 为 True/False 的二维数组"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0   # 避免除0
    return intersection / union


@torch.no_grad()
def calculate_iou_trajoffset(gt_video_path:str,
                  sample_videos_path:str,
                  modules_config: dict,
                  device,
                  width=832,
                  height=480,
                  video_cache_dir = "./cache"
                  ):

    model = modules_config["model"]
    processor = modules_config["processor"]



    #init
    cur_name = os.path.basename(gt_video_path).split(".")[0]
    gt_mask_path = os.path.join(os.path.dirname(gt_video_path),"mask_object_1.mp4")
    gt_mask_frames = load_frames(gt_mask_path,step=2)
    gt_mask_save_path = os.path.join(video_cache_dir,"gt_mask.mp4")
    export_to_video(gt_mask_frames,gt_mask_save_path,fps=15, macro_block_size=4)

    # get gt_mask_traj
    gt_mask_traj = []
    gt_mask_numpy_list = []
    coords_list = get_trajectory_from_mask(gt_mask_save_path)
    print("============= old method coords_list =========")
    print(coords_list)
    for mask in gt_mask_frames:
        center_coord,mask_numpy = get_coords_from_mask(mask)
        gt_mask_traj.append(center_coord)
        gt_mask_numpy_list.append(mask_numpy)
    print("============= gt_mask_traj ============")
    print(gt_mask_traj)

    # extract mask video for sample video
    sample_mask_save_path = os.path.join(video_cache_dir,f"{cur_name}_mask.mp4")
    frames1 = export_mask_from_mp4(model,processor,sample_videos_path,gt_mask_traj[0],sample_mask_save_path,device)

    # load sample_mask
    sample_mask_frames = load_frames(sample_mask_save_path,step=1)

    # get sample traj and mask_numpy
    sample_mask_traj = []
    sample_mask_numpy_list = []
    for mask in sample_mask_frames:
        center_coord,mask_numpy = get_coords_from_mask(mask)
        sample_mask_traj.append(center_coord)
        sample_mask_numpy_list.append(mask_numpy)
    print("============= sample traj ============")
    print(sample_mask_traj)

    # iou
    ious = []
    for m1,m2 in zip(gt_mask_numpy_list,sample_mask_numpy_list):
        iou = compute_iou(m1,m2)
        ious.append(iou)
    loss_iou = sum(ious) / len(ious)

    # trajoffset
    loss_traj_offset = position_loss(gt_mask_traj,sample_mask_traj,None,832,480)

    # # clear
    # os.remove(sample_mask_save_path)

    return loss_iou,loss_traj_offset

