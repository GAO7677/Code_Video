import torch, torchvision
from torch.utils.data import Dataset
import json
import os
import imageio
import random
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
import numpy as np
import random
import glob


def _open_video_reader(video_path):
    try:
        return imageio.get_reader(video_path, format="FFMPEG")
    except Exception:
        return imageio.get_reader(video_path)


def _safe_video_length(reader):
    try:
        frame_count = int(reader.count_frames())
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    try:
        frame_count = int(reader.get_length())
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    fps = meta.get("fps")
    duration = meta.get("duration")
    if fps and duration:
        estimated = int(round(float(fps) * float(duration)))
        if estimated > 0:
            return estimated

    try:
        frame_count = sum(1 for _ in reader)
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    raise RuntimeError("Unable to determine video frame count")


class WanV2V5BDataset(Dataset):
    def __init__(
        self, 
        json_path = "data/data.jsonl", 
        width = 1280, 
        height = 704,
        num_frames = 49,
        start_max = None,
        need_mask = True,
        data_repeat = 1,
    ):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.start_max = start_max
        self.data_list = []
        self.need_mask = need_mask
        if ".jsonl" in json_path:
            self.data_list = self.read_jsonl(json_path)
        elif ".json" in json_path:
            self.data_list = self.read_json(json_path)
        else:
            for file in os.listdir(json_path):
                path = os.path.join(json_path,file)
                if ".jsonl" in path:
                    self.data_list += self.read_jsonl(path)
                else:
                    self.data_list += self.read_json(path)

        self.data_list = self.data_list * data_repeat  # 或 self.data_list *= self.data_repeat
        print(f"Total Data Length : {len(self.data_list)}")
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def read_jsonl(self,json_path) -> list:
        data = []
        with open(json_path, "r") as f:
            for idx, line in enumerate(f):
                current_data = json.loads(line.strip())
                text = current_data["caption"]
                # text = current_data["qwen_omni_caption"]
                path = current_data["path"]
                data.append({"text":text,
                            "path":path})    
        return data    
    
    def read_json(self,json_path) -> list:
        data = []
        with open(json_path, "r") as f:
            datas = json.load(f)  
        for current_data in datas:
            text = current_data["captions"]
            # text = current_data["qwen_omni_caption"]
            path = os.path.join(self.video_dir, current_data["video_name"])
            data.append({"text":text,
                        "path":path})   
        return data    

    def __getitem__(self, idx):
        while True:
            try:
                text = self.data_list[idx]["text"]
                video_path = self.data_list[idx]["path"]
                frames = []

                # get frames
                with _open_video_reader(video_path) as reader:
                    mp4_frames = _safe_video_length(reader)
                    if mp4_frames < self.num_frames:
                        raise Exception(f"mp4_frames < self.num_frames,{mp4_frames}<{self.num_frames}")

                    fps = self.data_list[idx]["fps"] if "fps" in self.data_list[idx].keys() else reader.get_meta_data()['fps']

                    sample_step = 2 if round(fps) == 60 else 1 # 2

                    
                    if mp4_frames // sample_step < self.num_frames:
                        raise Exception(f"mp4_frames // sample_step < self.num_frames,{mp4_frames} // {sample_step}<{self.num_frames}")
                    
                    assert mp4_frames // sample_step >= self.num_frames , "length error"

                    total_num_frames = self.num_frames * sample_step
                    start_max = min(mp4_frames - total_num_frames+1, self.start_max) if self.start_max is not None else mp4_frames - total_num_frames+1
                    start = np.random.randint(low=0, high=start_max)
                    idxs = np.arange(start, start + total_num_frames, sample_step)

                    for frame_id in idxs:
                        if len(frames) == self.num_frames:
                            break
                        frame = reader.get_data(frame_id)
                        frame = Image.fromarray(frame)
                        frame = self.crop_and_resize(frame,self.height,self.width)
                        frames.append(frame)

                # get mask
                if self.need_mask:
                    mask_frames_1 = []
                    mask_frames_2 = []
                    video_dir = os.path.dirname(video_path)
                    mask_path1 = os.path.join(video_dir, "mask_object_1.mp4")
                    mask_path2 = os.path.join(video_dir, "mask_object_2.mp4")

                    with _open_video_reader(mask_path1) as m_reader:
                        for frame_id in idxs:
                            if len(mask_frames_1) == self.num_frames:
                                break
                            frame = m_reader.get_data(frame_id)
                            frame = Image.fromarray(frame)
                            frame = self.crop_and_resize(frame,self.height,self.width)
                            mask_frames_1.append(frame)
                    
                    with _open_video_reader(mask_path2) as m_reader:
                        for frame_id in idxs:
                            if len(mask_frames_2) == self.num_frames:
                                break
                            frame = m_reader.get_data(frame_id)
                            frame = Image.fromarray(frame)
                            frame = self.crop_and_resize(frame,self.height,self.width)
                            mask_frames_2.append(frame)                    
                
                    return video_path,text,frames,mask_frames_1,mask_frames_2
                else:
                    return video_path,text,frames
            except Exception as e:
                print(video_path,e)
                idx = random.randint(0, len(self.data_list) - 1)


    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        data = {}
        data["path"] = [d[0] for d in batch]
        data["text"] = [d[1] for d in batch]
        data["video"] = [d[2] for d in batch]
        if len(batch[0]) == 5:
            data["mask1"] = [d[3] for d in batch]
            data["mask2"] = [d[4] for d in batch]
        return data


class WanV2V5BInferDataset(Dataset):
    def __init__(
        self, 
        video_path = "/mnt/robby-b1/common/datasets/0807trans/tanshuai/wisa-80k-reli.jsonl", 
        width = 1280, 
        height = 704,
        num_frames = 49,
        standard_fps = 15,
        start_max = 8,
    ):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.standard_fps = standard_fps
        self.start_max = start_max
        self.data_list = []
        if ".jsonl" in video_path:
            self.data_list = self.read_jsonl(video_path)
        elif ".json" in video_path:
            self.data_list = self.read_json(video_path)
        elif ".mp4" in video_path:
            cur_data = {"text":"The video shows rigid body motion.",
                        "path":video_path}
            self.data_list.append(cur_data)
        else:
            mp4_files = glob.glob(os.path.join(video_path, "**", "*.mp4"), recursive=True)
            mp4_files = [x for x in mp4_files if "mask" not in x]
            for file in mp4_files:
                cur_data = {"text":"The video shows rigid body motion.",
                            "path":file}
                self.data_list.append(cur_data)

        print(f"Total Data Length : {len(self.data_list)}")
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def read_jsonl(self,json_path) -> list:
        data = []
        with open(json_path, "r") as f:
            for idx, line in enumerate(f):
                current_data = json.loads(line.strip())
                text = current_data["caption"]
                # text = current_data["qwen_omni_caption"]
                path = current_data["path"]
                data.append({"text":text,
                            "path":path})    
        return data    
    
    def read_json(self,json_path) -> list:
        data = []
        with open(json_path, "r") as f:
            datas = json.load(f)  
        for current_data in datas:
            text = current_data["captions"]
            # text = current_data["qwen_omni_caption"]
            path = os.path.join(self.video_dir, current_data["video_name"])
            data.append({"text":text,
                        "path":path})   
        return data    

    def __getitem__(self, idx):
        while True:
            try:
                text = self.data_list[idx]["text"]
                video_path = self.data_list[idx]["path"]
                frames = []
                with _open_video_reader(video_path) as reader:
                    mp4_frames = _safe_video_length(reader)
                    if mp4_frames < self.num_frames:
                        raise Exception(f"mp4_frames < self.num_frames,{mp4_frames}<{self.num_frames}")

                    fps = self.data_list[idx]["fps"] if "fps" in self.data_list[idx].keys() else reader.get_meta_data()['fps']

                    sample_step = 2 if round(fps) == 60 else 1 # 2

                    
                    if mp4_frames // sample_step < self.num_frames:
                        raise Exception(f"mp4_frames // sample_step < self.num_frames,{mp4_frames} // {sample_step}<{self.num_frames}")
                    
                    assert mp4_frames // sample_step >= self.num_frames , "length error"

                    total_num_frames = self.num_frames * sample_step
                    start_max = min(mp4_frames - total_num_frames+1, self.start_max) if self.start_max is not None else mp4_frames - total_num_frames+1
                    start = np.random.randint(low=0, high=start_max)
                    idxs = np.arange(start, start + total_num_frames, sample_step)

                    for frame_id in idxs:
                        if len(frames) == self.num_frames:
                            break
                        frame = reader.get_data(frame_id)
                        frame = Image.fromarray(frame)
                        frame = self.crop_and_resize(frame,self.height,self.width)
                        frames.append(frame)
                 
                    
                return text,frames,video_path
            except Exception as e:
                print(video_path,e)
                idx = random.randint(0, len(self.data_list) - 1)


    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        data = {}
        data["text"] = [d[0] for d in batch]
        data["video"] = [d[1] for d in batch]
        return data
