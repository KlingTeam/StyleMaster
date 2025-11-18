import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoStyleMasterPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from diffsynth.models.kolors_text_encoder import RMSNorm

from styleproj import Processor, StyleModel
from diffsynth.models.wan_video_dit import WanControlNet

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False, generate_control=False):
        metadata = pd.read_csv(metadata_path)
        self.path = metadata["video_path"].to_list()
        self.text = metadata["caption"].to_list()
        
        # 添加refvideo读取
        if "refvideo" in metadata.columns:
            self.refvideo_path = metadata["refvideo"].to_list()
        else:
            self.refvideo_path = [None] * len(self.path)
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.generate_control = generate_control
            
        # 原始视频处理pipeline
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 控制信号处理pipeline（下采样8倍→上采样→灰度）
        if self.generate_control:
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


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, refvideo_path=None):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        control_frames = []
        first_frame = None
        
        # 如果有refvideo，打开refvideo reader
        ref_reader = None
        if self.generate_control and refvideo_path and not pd.isna(refvideo_path):
            try:
                ref_reader = imageio.get_reader(refvideo_path)
            except:
                ref_reader = None
        
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            
            if first_frame is None:
                first_frame = np.array(frame)
            
            # 处理原始帧
            processed_frame = self.frame_process(frame)
            frames.append(processed_frame)
            
            # 如果需要生成控制信号
            if self.generate_control:
                # if ref_reader is not None:
                #     # 从refvideo读取对应帧
                #     try:
                #         ref_frame = ref_reader.get_data(start_frame_id + frame_id * interval)
                #         ref_frame = Image.fromarray(ref_frame)
                #         ref_frame = self.crop_and_resize(ref_frame)
                #         control_frame = self.control_process(ref_frame)
                #     except:
                #         # 回退到原始帧
                #         control_frame = self.control_process(frame)
                # else:
                #     # 使用原始帧
                #     control_frame = self.control_process(frame)
                control_frame = self.control_process(frame)
                control_frames.append(control_frame)
        
        reader.close()
        if ref_reader is not None:
            ref_reader.close()
        
        # 填充到81帧
        if max_num_frames != 1:
            while len(frames) < 81:
                frames.append(frames[-1])
                if self.generate_control:
                    control_frames.append(control_frames[-1])
        
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        result = {"video": frames}
        
        if self.generate_control:
            control_frames = torch.stack(control_frames, dim=0)
            control_frames = rearrange(control_frames, "T C H W -> C T H W")
            result["control_video"] = control_frames

        if self.is_i2v:
            result["first_frame"] = first_frame
            
        return result


    def load_video(self, file_path, refvideo_path=None):
        start_frame_id = 0
        return self.load_frames_using_imageio(
            file_path, self.max_num_frames, start_frame_id, 
            self.frame_interval, self.num_frames, refvideo_path
        )
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path, refvideo_path=None):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        
        # 处理原始图像
        processed_frame = self.frame_process(frame)
        processed_frame = rearrange(processed_frame, "C H W -> C 1 H W")
        
        result = {"video": processed_frame}
        
        # 如果需要生成控制信号
        if self.generate_control:
            if refvideo_path and not pd.isna(refvideo_path):
                try:
                    if self.is_image(refvideo_path):
                        ref_frame = Image.open(refvideo_path).convert("RGB")
                    else:
                        ref_reader = imageio.get_reader(refvideo_path)
                        ref_frame = Image.fromarray(ref_reader.get_data(0))
                        ref_reader.close()
                    ref_frame = self.crop_and_resize(ref_frame)
                    control_frame = self.control_process(ref_frame)
                except:
                    control_frame = self.control_process(frame)
            else:
                control_frame = self.control_process(frame)
            
            control_frame = rearrange(control_frame, "C H W -> C 1 H W")
            result["control_video"] = control_frame
            
        return result


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        refvideo_path = self.refvideo_path[data_id]
        
        while True:
            try:
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    result = self.load_image(path, refvideo_path)
                else:
                    result = self.load_video(path, refvideo_path)
                
                # 构建返回数据
                data = {
                    "text": text, 
                    "video": result["video"], 
                    "path": path
                }
                
                if "control_video" in result:
                    data["control_video"] = result["control_video"]
                    
                if "first_frame" in result:
                    data["first_frame"] = result["first_frame"]
                    
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), generate_control=False):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoStyleMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.generate_control = generate_control
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            pth_path = path + ".tensors.pth"
            # if not os.path.exists(pth_path):
            if True:
                # prompt
                prompt_emb = self.pipe.encode_prompt(text)
                
                # 编码原始视频
                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                
                # 编码控制视频（如果存在）
                control_latents = None
                if self.generate_control and "control_video" in batch:
                    control_video = batch["control_video"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                    control_latents = self.pipe.encode_video(control_video, **self.tiler_kwargs)[0]
                
                # image embedding
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                
                # 保存数据
                data = {
                    "latents": latents, 
                    "prompt_emb": prompt_emb, 
                    "image_emb": image_emb
                }
                
                if control_latents is not None:
                    data["control_latents"] = control_latents
                
                torch.save(data, pth_path)
                print(f"Saved: {pth_path}")
                
            else:
                print(f"File {pth_path} already exists, skipping.")




class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, training_mode="style"):
        metadata = pd.read_csv(metadata_path)
        self.path = metadata["video_path"]
        self.full_style_path = metadata["style"]
        self.training_mode = training_mode
        
        valid_indices = [index for index, i in enumerate(self.path) if os.path.exists(i + ".tensors.pth")]

        # 更新路径
        self.path = [self.path[index] + ".tensors.pth" for index in valid_indices]
        self.style_path = [self.full_style_path[index] for index in valid_indices]
        
        # 对于controlnet模式，验证是否有control_latents
        if training_mode == "controlnet" or training_mode == "style_w_controlnet":
            valid_control_indices = []
            for i, path in enumerate(self.path):
                try:
                    data = torch.load(path, weights_only=False, map_location="cpu")
                    if "control_latents" in data:
                        valid_control_indices.append(i)
                except:
                    continue
            
            if len(valid_control_indices) == 0:
                raise ValueError("No control_latents found in tensor files. Please regenerate data with --generate_control flag.")
            
            # 只保留有控制数据的文件
            self.path = [self.path[i] for i in valid_control_indices]
            self.style_path = [self.style_path[i] for i in valid_control_indices]
            print(f"Found {len(self.path)} files with control data for ControlNet training.")
        
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index):
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path)
                path_tgt = self.path[data_id]
                
                data_tgt = torch.load(path_tgt, weights_only=False, map_location="cpu")

                data['latents'] = data_tgt['latents']
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb'] = data_tgt.get('image_emb', {})
                
                # 无论什么模式都需要style_img_path
                style_image_path = self.style_path[data_id]
                data['style_img_path'] = style_image_path
                
                if self.training_mode == "controlnet" or self.training_mode == "style_w_controlnet":
                    # 直接从tensor文件中加载控制latents
                    data['control_latents'] = data_tgt['control_latents']
                
                break
            except Exception as e:
                print(f"Error loading data: {e}")
                index = random.randrange(len(self.path))
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        training_mode="style",  # 'style' 或 'controlnet'
        controlnet_num_layer_stride=2,  # ControlNet层间隔
        learning_rate=1e-5,
        use_gradient_checkpointing=True, 
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        control_resume_ckpt_path=None,
    ):
        super().__init__()
        self.training_mode = training_mode
        
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models(dit_path)
        
        self.pipe = WanVideoStyleMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # 总是设置style功能
        self._setup_style_training()
        
        # 如果是controlnet模式，额外设置controlnet
        if training_mode == "controlnet" or self.training_mode == "style_w_controlnet":
            self._setup_controlnet_training(controlnet_num_layer_stride)

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            if control_resume_ckpt_path is not None:
                control_state_dict = torch.load(control_resume_ckpt_path, map_location="cpu")
            if training_mode == "style":
                self.pipe.dit.load_state_dict(state_dict, strict=False)
            elif training_mode == "controlnet" or self.training_mode == "style_w_controlnet":
                # 先加载主模型权重
                self.pipe.dit.load_state_dict(state_dict, strict=False)
                # 再加载controlnet权重（如果存在）
                if control_resume_ckpt_path is not None:
                    control_state_dict = torch.load(control_resume_ckpt_path, map_location="cpu")
                    self.controlnet.load_state_dict(control_state_dict, strict=False)

        self.freeze_parameters()
        self._setup_trainable_parameters()
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
    def _setup_style_training(self):
        """设置style训练模式 - 总是需要"""
        # Initialize the style processor and model
        self.processor = Processor().eval()
        dim = self.pipe.dit.blocks[0].cross_attn.q.weight.shape[0]
        
        self.pipe.dit.style_model = StyleModel()
        dim_s = self.pipe.dit.style_model.cross_attention_dim
        
        for block in self.pipe.dit.blocks:
            block.cross_attn.k_img = nn.Linear(dim_s, dim)
            block.cross_attn.v_img = nn.Linear(dim_s, dim)
            block.cross_attn.norm_k_img = RMSNorm(dim)
            # 初始化为0
            block.cross_attn.k_img.weight.data.zero_()
            block.cross_attn.k_img.bias.data.zero_()
            block.cross_attn.v_img.weight.data.zero_()
            block.cross_attn.v_img.bias.data.zero_()
            block.cross_attn.norm_k_img.weight.data.zero_()
    
    def _setup_controlnet_training(self, num_layer_stride):
        """设置controlnet训练模式 - 仅在controlnet模式下需要"""
        # 获取WanModel的配置
        config = {
            'dim': self.pipe.dit.dim,
            'in_dim': self.pipe.dit.patch_embedding.in_channels,
            'ffn_dim': self.pipe.dit.blocks[0].ffn_dim,
            'text_dim': self.pipe.dit.text_embedding[0].in_features,
            'freq_dim': self.pipe.dit.freq_dim,
            'eps': 1e-6,
            'patch_size': self.pipe.dit.patch_size,
            'num_heads': self.pipe.dit.blocks[0].num_heads,
            'num_layers': len(self.pipe.dit.blocks),
            'has_image_input': self.pipe.dit.has_image_input,
            'num_layer_stride': num_layer_stride,
        }
        
        # 创建ControlNet并从WanModel初始化
        self.controlnet = WanControlNet(**config)
        self.controlnet.load_from_wan_state_dict(self.pipe.dit.state_dict())
        
    def _setup_trainable_parameters(self):
        """设置可训练参数"""
        if self.training_mode == "style" or self.training_mode == "style_w_controlnet":
            # Style模式：只训练style_model和相关attention层
            for name, module in self.pipe.denoising_model().named_modules():
                if any(keyword in name for keyword in ["style_model", "_img"]):
                    print(f"Trainable: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
            if self.training_mode == "style_w_controlnet":
                for param in self.controlnet.parameters():
                    param.requires_grad = True
        elif self.training_mode == "controlnet":
            # ControlNet模式：只训练controlnet，style部分保持冻结
            for param in self.controlnet.parameters():
                param.requires_grad = True
            print("ControlNet parameters set to trainable")

        # 计算可训练参数数量
        if self.training_mode == "style" or self.training_mode == "style_w_controlnet":
            trainable_params = 0
            seen_params = set()
            for name, module in self.pipe.denoising_model().named_modules():
                for param in module.parameters():
                    if param.requires_grad and param not in seen_params:
                        trainable_params += param.numel()
                        seen_params.add(param)
        else:
            trainable_params = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        
        print(f"Total number of trainable parameters: {trainable_params}")
        
    def freeze_parameters(self):
        """冻结参数"""
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        if hasattr(self, 'controlnet'):
            self.controlnet.requires_grad_(False)
            self.controlnet.train()

    def training_step(self, batch, batch_idx):
        if self.training_mode == "style":
            return self._style_training_step(batch, batch_idx)
        elif self.training_mode == "controlnet" or self.training_mode == "style_w_controlnet":
            return self._controlnet_training_step(batch, batch_idx)
    
    def _style_training_step(self, batch, batch_idx):
        """Style训练步骤 - 不使用controlnet"""
        # 1. Prepare Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"].to(self.device)

        # 2. Process Style Image
        ref_images = []
        for path in batch["style_img_path"]:
            img = Image.open(path)
            ref_images.append(img)
        image_embeds, image_embeds_ps = self.processor.process_images(ref_images)

        style_features = self.pipe.dit.style_model.project_image_embeddings(
            image_embeds, image_embeds_ps, prompt_emb["context"].squeeze(1)
        )

        # 3. Standard Denoising Loss Calculation
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # 4. Compute loss - 不传入control_state
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            style_feat=style_features, 
            **prompt_emb, 
            **extra_input, 
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def _controlnet_training_step(self, batch, batch_idx):
        """ControlNet训练步骤 - 直接在latent空间工作"""
        # 1. Prepare Data
        latents = batch["latents"].to(self.device)
        control_latents = batch["control_latents"].to(self.device)  # 控制latents
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"].to(self.device)

        # 2. Process Style Image (仍然需要style功能)
        ref_images = []
        for path in batch["style_img_path"]:
            img = Image.open(path)
            ref_images.append(img)
        image_embeds, image_embeds_ps = self.processor.process_images(ref_images)

        style_features = self.pipe.dit.style_model.project_image_embeddings(
            image_embeds, image_embeds_ps, prompt_emb["context"].squeeze(1)
        )

        # 3. 准备时间步 - 确保设备一致性
        self.pipe.device = self.device  # 确保pipe的device设置正确
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.device)

        # 4. 获取ControlNet输出 - 确保controlnet在正确设备上
        self.controlnet.to(self.device)
    
        control_input = control_latents
        # 直接使用控制latents，不需要decode
        control_state = self.controlnet(
            x=control_input,  # 直接使用latents
            timestep=timestep,  # 确保timestep在正确设备上
            context=prompt_emb["context"],
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )

        # 5. Standard Denoising Loss Calculation
        noise = torch.randn_like(latents)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # 6. 使用style_features和control_state进行预测
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            style_feat=style_features,  # 使用真实的style特征
            control_state=control_state,
            controlnet_conditioning_scale=1.0,
            **prompt_emb, 
            **extra_input, 
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.training_mode == "style":
            trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        elif self.training_mode == "style_w_controlnet":
            denoising_params = list(filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters()))
            controlnet_params = list(filter(lambda p: p.requires_grad, self.controlnet.parameters()))
            trainable_modules = denoising_params + controlnet_params
        else:
            trainable_modules = filter(lambda p: p.requires_grad, self.controlnet.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        
        if self.training_mode == "style":
            state_dict = self.pipe.denoising_model().state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
        elif self.training_mode == "style_w_controlnet":
            # 保存 denoising model
            denoising_state_dict = self.pipe.denoising_model().state_dict()
            torch.save(denoising_state_dict, os.path.join(checkpoint_dir, f"denoising_step{current_step}.ckpt"))
            
            # 保存 controlnet
            controlnet_state_dict = self.controlnet.state_dict()
            torch.save(controlnet_state_dict, os.path.join(checkpoint_dir, f"controlnet_step{current_step}.ckpt"))
        else:
            state_dict = self.controlnet.state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))

def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="style",
        choices=["style", "controlnet", "style_w_controlnet"],
        help="Training mode: 'style' for style training, 'controlnet' for controlnet training.",
    )
    parser.add_argument(
        "--generate_control",
        default=False,
        action="store_true",
        help="Whether to generate control signals during data processing.",
    )
    parser.add_argument(
        "--controlnet_num_layer_stride",
        type=int,
        default=2,
        help="ControlNet layer stride.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--control_resume_ckpt_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        generate_control=args.generate_control
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        generate_control=args.generate_control
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)

from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    """
    自定义的数据整理函数。
    """
    keys = batch[0].keys()
    collated_batch = {}

    for key in keys:
        if key in ['style_img_path']:
            collated_batch[key] = [item[key] for item in batch]
        else:
            data_list = [item[key] for item in batch]
            collated_batch[key] = default_collate(data_list)
            
    return collated_batch


def train(args):
    try:
        dataset = TensorDataset(
            args.dataset_path,
            os.path.join(args.dataset_path, args.metadata_file_name),
            steps_per_epoch=args.steps_per_epoch,
            training_mode=args.training_mode,
        )
        print("Dataset length:", len(dataset))
    except Exception as e:
        print("Error in dataset creation or access:", e)
        return
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    try:
        for batch in dataloader:
            break
        print("DataLoader can be iterated successfully.")
    except Exception as e:
        print("Error during iteration:", e)
        return
        
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        training_mode=args.training_mode,
        controlnet_num_layer_stride=args.controlnet_num_layer_stride,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        control_resume_ckpt_path=args.control_resume_ckpt_path,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name=f"wan_{args.training_mode}",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1,
                                                        every_n_train_steps=500)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)