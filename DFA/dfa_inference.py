#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import json
import argparse
import cv2
from dataclasses import dataclass
from torchvision import transforms

USER = os.environ.get('USER', 'xz223')
TMP_BASE = f'/tmp/{USER}/fact_inference'
os.makedirs(TMP_BASE, exist_ok=True)

os.environ['HF_HOME'] = f'{TMP_BASE}/huggingface'
os.environ['TRANSFORMERS_CACHE'] = f'{TMP_BASE}/huggingface'
os.environ['MODELSCOPE_CACHE'] = f'{TMP_BASE}/modelscope'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_DIR = Path('/vol/bitbucket/xz223/dlenv/PPG')
sys.path.append(str(PROJECT_DIR))

from face_adapter import Face_Transformer, Face_Prj_Resampler
from face_preprocess import preprocess

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except ImportError as e:
    logger.error(f"Failed to import ModelScope: {e}")
    sys.exit(1)


@dataclass
class InferenceConfig:
    checkpoint_path: str = '/vol/bitbucket/xz223/dlenv/PPG/checkpoints/adapter_dfa.ckpt'
    output_dir: str = '/vol/bitbucket/xz223/dlenv/PPG/inference_results'
    transface_weight: str = '/vol/bitbucket/xz223/dlenv/PPG/weights/ms1mv2_model_TransFace_S.pt'
    sd_model_id: str = 'runwayml/stable-diffusion-v1-5'
    
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    face_strength: float = 1.0
    height: int = 512
    width: int = 512



def detect(image, face_detection):
    result_det = face_detection(image)
    
    if result_det is None or 'scores' not in result_det or not result_det['scores']:
        raise ValueError("Face detection failed")
        
    confs = result_det['scores']
    idx = np.argmax(confs)
    
    if confs[idx] < 0.5:
        raise ValueError(f"Face detection confidence too low: {confs[idx]}")
    
    if 'keypoints' not in result_det or len(result_det['keypoints']) <= idx:
        raise ValueError("No face keypoints detected")
        
    pts = result_det['keypoints'][idx]
    if pts is None:
        raise ValueError("Keypoints is None")
        
    points_vec = np.array(pts).reshape(5, 2)
    return points_vec


def get_mask_head(result):
    if result is None:
        raise ValueError("Segmentation result is None")
        
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    
    img_shape = masks[0].shape
    mask_hair = np.zeros(img_shape)
    mask_face = np.zeros(img_shape)
    mask_human = np.zeros(img_shape)
    
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    
    if np.sum(mask_face) == 0:
        raise ValueError("No face mask detected")
    
    mask_head = np.clip(mask_hair + mask_face, 0, 1)
    
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1)
    
    if np.sum(mask_human) > 0:
        mask_head = mask_head * mask_human
    
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No mask contours found")
        
    areas = [cv2.contourArea(contour) for contour in contours]
    max_idx = np.argmax(areas)
    
    mask_head = np.zeros(img_shape).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255
    
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2)
    
    return mask_head


def align(image, points_vec):
    if points_vec is None:
        raise ValueError("Keypoints is None")
        
    img_array = np.array(image)[:,:,::-1]
    
    warped = preprocess(
        img_array, 
        bbox=None, 
        landmark=points_vec, 
        image_size='112, 112'
    )
    
    if warped is None:
        raise ValueError("Face alignment failed")
    
    aligned_face = Image.fromarray(warped[:,:,::-1])
    return aligned_face


def face_image_preprocess(image, segmentation_pipeline, face_detection):
    result = segmentation_pipeline(image)
    mask_head = get_mask_head(result)
    
    image_array = np.array(image)
    image_masked_array = (image_array * mask_head).astype(np.uint8)
    image_masked = Image.fromarray(image_masked_array)
    
    points_vec = detect(image_masked, face_detection)
    aligned_image = align(image_masked, points_vec)
    
    return aligned_image



class SelfAttention(nn.Module):
    def __init__(self, query_dim, inner_dim, dropout=0.):
        super().__init__()
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        if hasattr(attn, 'head_to_batch_dim'):
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * (q.shape[-1] ** -0.5)
            attn_weights = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn_weights, v)
            
            out = attn.batch_to_head_dim(out)
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * (q.shape[-1] ** -0.5)
            attn_weights = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn_weights, v)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., glu=True):
        super().__init__()
        inner_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2 if glu else inner_dim, bias=False),
            nn.GELU() if not glu else GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )
    
    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FaceAttnProcessor(nn.Module):
    """Face attention processor matching training code"""
    
    def __init__(self, hidden_size, cross_attention_dim=768, scale=1.0, ln=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.ln = ln
        self.num_ims = 1
        
        self.attn = SelfAttention(hidden_size, hidden_size)
        self.ff = FeedForward(hidden_size)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual0 = hidden_states
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        if encoder_hidden_states is not None:
            end_pos = encoder_hidden_states.shape[1] - 16 * self.num_ims
            text_hidden_states = encoder_hidden_states[:, :end_pos, :]
            face_hidden_states = encoder_hidden_states[:, end_pos:, :]
            
            if hasattr(attn, 'norm_cross') and attn.norm_cross:
                text_hidden_states = attn.norm_encoder_hidden_states(text_hidden_states)
        else:
            text_hidden_states = encoder_hidden_states
            face_hidden_states = None
        
        if face_hidden_states is not None:
            N_visual = hidden_states.shape[1]
            
            hidden_states_ln = self.norm1(hidden_states)
            face_hidden_states_ln = self.norm1(face_hidden_states)
            combined = torch.cat([hidden_states_ln, face_hidden_states_ln], dim=1)
            
            attn_out = self.attn(combined, attn)[:, :N_visual, :]
            hidden_states = hidden_states + self.scale * torch.tanh(self.alpha_attn) * attn_out
            
            hidden_states_ln = self.norm2(hidden_states)
            ff_out = self.ff(hidden_states_ln)
            hidden_states = hidden_states + self.scale * torch.tanh(self.alpha_dense) * ff_out
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        residual = hidden_states
        
        if self.ln is not None:
            hidden_states = self.ln(hidden_states)
        
        if hasattr(attn, 'spatial_norm') and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if text_hidden_states is None else text_hidden_states.shape
        )
        
        if hasattr(attn, 'prepare_attention_mask'):
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if hasattr(attn, 'group_norm') and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(text_hidden_states if text_hidden_states is not None else hidden_states)
        value = attn.to_v(text_hidden_states if text_hidden_states is not None else hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if hasattr(attn, 'residual_connection') and attn.residual_connection:
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states + residual - residual0
        
        if hasattr(attn, 'rescale_output_factor'):
            hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class FACTInferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = device
        
        self._load_models()
        self._load_checkpoint()
        self._set_inference_mode()
    
    def _load_models(self):
        from diffusers import (
            AutoencoderKL,
            UNet2DConditionModel,
            DDIMScheduler
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        
        logger.info("Loading Stable Diffusion components...")
        
        self.vae = AutoencoderKL.from_pretrained(
            self.config.sd_model_id,
            subfolder="vae",
            torch_dtype=torch.float16,
            cache_dir=f"{TMP_BASE}/models"
        ).to(self.device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.sd_model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
            cache_dir=f"{TMP_BASE}/models"
        ).to(self.device)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.sd_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            cache_dir=f"{TMP_BASE}/models"
        ).to(self.device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.sd_model_id,
            subfolder="tokenizer",
            cache_dir=f"{TMP_BASE}/models"
        )
        
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config.sd_model_id,
            subfolder="scheduler",
            cache_dir=f"{TMP_BASE}/models"
        )
        
        # Face components
        self.face_transformer = Face_Transformer(weight=self.config.transface_weight)
        self.face_transformer.eval().to(self.device)
        
        self.face_projector = Face_Prj_Resampler(
            dim=1024,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=512,
            output_dim=768,
            ff_mult=4
        ).to(self.device)
        
        # ModelScope components
        logger.info("Loading ModelScope components...")
        self.face_detection = pipeline(
            task=Tasks.face_detection,
            model='damo/cv_resnet50_face-detection_retinaface',
            model_revision='v1.0.0',
            cache_dir=f'{TMP_BASE}/modelscope'
        )
        self.segmentation_pipeline = pipeline(
            task=Tasks.image_segmentation,
            model='damo/cv_resnet101_image-multiple-human-parsing',
            model_revision='v1.0.0',
            cache_dir=f'{TMP_BASE}/modelscope'
        )
        
        logger.info("Models loaded successfully")
    
    def _load_checkpoint(self):
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        logger.info(f"Loading checkpoint from: {self.config.checkpoint_path}")
        
        # Load face projector weights
        projector_state = {}
        for key, value in checkpoint.items():
            if key.startswith('local_fac_prj.'):
                new_key = key.replace('local_fac_prj.', '')
                projector_state[new_key] = value
        
        if projector_state:
            self.face_projector.load_state_dict(projector_state, strict=False)
            logger.info(f"Loaded face projector: {len(projector_state)} parameters")
        
        # Inject and load processors
        self._inject_processors(checkpoint)
        
        logger.info("Checkpoint loaded successfully")
    
    def _inject_processors(self, checkpoint):
        from diffusers.models.attention_processor import AttnProcessor
        
        layer_norms = {}
        
        # Replace layer norms
        for i in range(3):
            for j in range(2):
                if hasattr(self.unet.down_blocks[i], 'attentions'):
                    key = f'down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn2.processor'
                    ln = self.unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2
                    layer_norms[key] = ln
                    self.unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        for i in range(3):
            for j in range(3):
                if hasattr(self.unet.up_blocks[i+1], 'attentions'):
                    key = f'up_blocks.{i+1}.attentions.{j}.transformer_blocks.0.attn2.processor'
                    ln = self.unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2
                    layer_norms[key] = ln
                    self.unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        if hasattr(self.unet.mid_block, 'attentions'):
            key = 'mid_block.attentions.0.transformer_blocks.0.attn2.processor'
            ln = self.unet.mid_block.attentions[0].transformer_blocks[0].norm2
            layer_norms[key] = ln
            self.unet.mid_block.attentions[0].transformer_blocks[0].norm2 = Identity()
        
        # Set attention processors
        attn_procs = {}
        self.processors = {}
        
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                attn_procs[name] = AttnProcessor()
                continue
            
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                processor = FaceAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=self.config.face_strength,
                    ln=layer_norms.get(name)
                ).to(self.device)
                
                # Load processor weights from checkpoint
                processor_state = {}
                for key, value in checkpoint.items():
                    if key.startswith(f'{name}.'):
                        param_name = key.replace(f'{name}.', '')
                        processor_state[param_name] = value
                
                if processor_state:
                    processor.load_state_dict(processor_state, strict=False)
                
                attn_procs[name] = processor
                self.processors[name] = processor
        
        self.unet.set_attn_processor(attn_procs)
        logger.info(f"Injected {len(self.processors)} face attention processors")
    
    def _set_inference_mode(self):
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.face_transformer.eval()
        self.face_projector.eval()
        
        if torch.cuda.is_available():
            self.vae = self.vae.half()
            self.unet = self.unet.half()
            self.text_encoder = self.text_encoder.half()
    
    def preprocess_face(self, face_image_path: str) -> torch.Tensor:
        if isinstance(face_image_path, str):
            image = Image.open(face_image_path).convert('RGB')
        else:
            image = face_image_path
        
        # Resize to 512x512 as in training
        image_resized = image.resize((512, 512), Image.LANCZOS)
        
        # Apply ModelScope preprocessing
        aligned_face = face_image_preprocess(
            image_resized,
            self.segmentation_pipeline,
            self.face_detection
        )
        
        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        face_tensor = transform(aligned_face).unsqueeze(0).to(self.device)
        
        return face_tensor
    
    @torch.no_grad()
    def generate(
        self,
        face_image_path: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted",
        num_images: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        
        # Preprocess face
        face_tensor = self.preprocess_face(face_image_path)
        
        # Extract face features
        local_features, _ = self.face_transformer(face_tensor)
        face_embeddings = self.face_projector(local_features)
        face_embeddings = face_embeddings.to(self.unet.dtype)
        
        # Encode text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        # Negative prompt
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]
        
        # Prepare embeddings
        batch_size = num_images
        
        face_embeddings = face_embeddings.repeat(batch_size, 1, 1)
        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)
        uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
        
        # Combine embeddings
        cond_embeddings = torch.cat([text_embeddings, face_embeddings], dim=1)
        uncond_embeddings = torch.cat([uncond_embeddings, face_embeddings], dim=1)
        
        # Set random seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Initialize latents
        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            self.config.height // 8,
            self.config.width // 8
        )
        
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=torch.float16
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings])
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        
        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        
        pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images


def main():
    parser = argparse.ArgumentParser(description='FACT Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', type=str, default='a professional portrait photo')
    parser.add_argument('--output-dir', type=str, default='./inference_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-images', type=int, default=1)
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    logger.info("Initializing FACT inference pipeline...")
    pipeline = FACTInferencePipeline(config)
    
    logger.info(f"Processing: {args.input}")
    
    results = pipeline.generate(
        face_image_path=args.input,
        prompt=args.prompt,
        negative_prompt="low quality, blurry",
        num_images=args.num_images,
        seed=args.seed
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(results):
        save_path = output_dir / f"result_{i}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")


if __name__ == '__main__':
    main()