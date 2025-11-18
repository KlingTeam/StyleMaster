import sys
import torch
import torch.nn as nn
from diffsynth import ModelManager, WanVideoStyleMasterPipeline, save_video, VideoData
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json
from diffsynth.models.kolors_text_encoder import RMSNorm

from styleproj import Processor, StyleModel
from diffsynth.models.wan_video_dit import WanControlNet


class TextVideoStyleDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, args, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.text = metadata["text"].to_list()
        self.style_path = metadata["style"].to_list()
        
        # 如果有control列，加载控制视频路径
        if "control" in metadata.columns:
            self.control_path = metadata["control"].to_list()
        else:
            self.control_path = [None] * len(self.text)
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.args = args
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 控制信号处理pipeline（下采样8倍→上采样→灰度）
        self.control_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height//8, width//8), antialias=True),  # 下采样8倍
            v2.Resize(size=(height, width), antialias=True),        # 上采样回原尺寸
            v2.Grayscale(num_output_channels=3),                    # 转为灰度但保持3通道
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        # 填充到81帧
        while len(frames) < 81:
            frames.append(frames[-1])

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    

    def load_video(self, file_path):
        start_frame_id = 0  # 推理时使用固定起始帧
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames

    def load_control_video(self, file_path):
        """加载控制视频"""
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.control_process)
        return frames


    def __getitem__(self, data_id):
        text = self.text[data_id]
        style = self.style_path[data_id]
        control = self.control_path[data_id]
        
        data = {"text": text, "style": style}
        
        # 如果有控制视频路径，加载控制视频
        if control is not None and os.path.exists(control):
            if self.is_image(control):
                # 如果是图像，转换为单帧视频
                frame = Image.open(control).convert("RGB")
                frame = self.crop_and_resize(frame)
                control_frame = self.control_process(frame)
                control_frame = rearrange(control_frame, "C H W -> C 1 H W")
                # 扩展到81帧
                control_frames = control_frame.repeat(1, 81, 1, 1)
                data["control"] = control_frames
            else:
                # 如果是视频
                control_video = self.load_control_video(control)
                if control_video is not None:
                    data["control"] = control_video
        
        return data
    

    def __len__(self):
        return len(self.style_path)

def parse_args():
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./example_test_data",
        help="The path of the Dataset.",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/stylemaster.ckpt",
        help="Path to save the model.",
    )
    
    parser.add_argument(
        "--controlnet_ckpt_path",
        type=str,
        default='checkpoints/controlnet.ckpt',
        help="Path to ControlNet checkpoint.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--style_cfg_scale",
        type=float,
        default=3,
    )
    
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1,
        help="ControlNet conditioning scale.",
    )
    
    parser.add_argument(
        "--controlnet_num_layer_stride",
        type=int,
        default=2,
        help="ControlNet layer stride.",
    )
    
    # 新增：ControlNet控制步数参数
    parser.add_argument(
        "--controlnet_guidance_start",
        type=float,
        default=0.0,
        help="ControlNet guidance start ratio (0.0-1.0). 0.0 means start from beginning.",
    )
    
    parser.add_argument(
        "--controlnet_guidance_end",
        type=float,
        default=0.7,
        help="ControlNet guidance end ratio (0.0-1.0). 1.0 means apply until the end.",
    )
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    # 1. Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoStyleMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. Initialize additional modules introduced in ReCamMaster
    processor = Processor().eval()
    dim = pipe.dit.blocks[0].cross_attn.q.weight.shape[0]
    pipe.dit.style_model = StyleModel()
    dim_s = pipe.dit.style_model.cross_attention_dim
    for block in pipe.dit.blocks:
        block.cross_attn.k_img = nn.Linear(dim_s, dim)
        block.cross_attn.v_img = nn.Linear(dim_s, dim)
        block.cross_attn.norm_k_img = RMSNorm(dim)

    # 3. Load style checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    for name in state_dict.keys():
        print(name)
    
    pipe.dit.load_state_dict(state_dict, strict=True)
    
    # 4. Initialize and load ControlNet if provided
    controlnet = None
    if args.controlnet_ckpt_path is not None:
        # 获取WanModel的配置
        config = {
            'dim': pipe.dit.dim,
            'in_dim': pipe.dit.patch_embedding.in_channels,
            'ffn_dim': pipe.dit.blocks[0].ffn_dim,
            'text_dim': pipe.dit.text_embedding[0].in_features,
            'freq_dim': pipe.dit.freq_dim,
            'eps': 1e-6,
            'patch_size': pipe.dit.patch_size,
            'num_heads': pipe.dit.blocks[0].num_heads,
            'num_layers': len(pipe.dit.blocks),
            'has_image_input': pipe.dit.has_image_input,
            'num_layer_stride': args.controlnet_num_layer_stride,
        }
        
        # 创建ControlNet并加载权重
        controlnet = WanControlNet(**config)
        controlnet_state_dict = torch.load(args.controlnet_ckpt_path, map_location="cpu")
        controlnet.load_state_dict(controlnet_state_dict, strict=True)
        controlnet.to("cuda")
        controlnet.to(dtype=torch.bfloat16)
        controlnet.eval()
        print("ControlNet loaded successfully")
        print(f"ControlNet guidance will be applied from step {args.controlnet_guidance_start:.1%} to {args.controlnet_guidance_end:.1%}")
    
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    output_dir = os.path.join(args.output_dir, 'style_controlnet' if controlnet is not None else 'style')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. Prepare test data
    dataset = TextVideoStyleDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # 6. Inference
    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["text"]
        target_style = batch["style"]

        # 处理style图像
        ref_images = []
        for path in target_style:
            img = Image.open(path)
            ref_images.append(img)
        image_embeds = processor.process_images(ref_images)
        
        # 处理控制信号
        control_video = None
        if "control" in batch and controlnet is not None:
            control_video = batch["control"].to(dtype=torch.bfloat16, device="cuda")
            print(f"Using control video with shape: {control_video.shape}")
            
        # 生成视频
        if controlnet is not None and control_video is not None:
            # 使用ControlNet生成
            video = pipe(
                prompt=target_text,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                target_style=image_embeds,
                control_video=control_video,
                controlnet=controlnet,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                controlnet_guidance_start=args.controlnet_guidance_start,  # 新增
                controlnet_guidance_end=args.controlnet_guidance_end,      # 新增
                cfg_scale=args.cfg_scale,
                style_cfg_scale=args.style_cfg_scale,
                num_inference_steps=50,
                seed=0, 
                tiled=True
            )
        else:
            # 仅使用style生成
            video = pipe(
                prompt=target_text,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                target_style=image_embeds,
                cfg_scale=args.cfg_scale,
                style_cfg_scale=args.style_cfg_scale,
                num_inference_steps=50,
                seed=0, 
                tiled=True
            )
            
        save_video(video, os.path.join(output_dir, f"video{batch_idx}.mp4"), fps=30, quality=5)
        print(f"Saved video {batch_idx}")