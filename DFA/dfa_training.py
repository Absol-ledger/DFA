#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import random
from dataclasses import dataclass, field
import argparse
from datetime import datetime
from collections import defaultdict

# Environment setup
USER = os.environ.get('USER', 'xz223')
SCRATCH_DIR = os.environ.get('TMPDIR', f'/tmp/{USER}')
TMP_BASE = f'{SCRATCH_DIR}/fact_training'
os.makedirs(TMP_BASE, exist_ok=True)

os.environ['HF_HOME'] = f'{TMP_BASE}/huggingface'
os.environ['TRANSFORMERS_CACHE'] = f'{TMP_BASE}/huggingface'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = Path('/vol/bitbucket/xz223/dlenv/PPG')
sys.path.append(str(PROJECT_DIR))

from face_adapter import Face_Transformer, Face_Prj_Resampler


@dataclass
class TrainingConfig:
    # Paths
    data_root: str = '/vol/bitbucket/xz223/dlenv/PPG/datasets'
    output_dir: str = f'{TMP_BASE}/outputs'
    checkpoint_dir: str = '/vol/bitbucket/xz223/dlenv/PPG/checkpoints'
    
    # Model configuration
    sd_model_id: str = 'runwayml/stable-diffusion-v1-5'
    transface_weight: str = '/vol/bitbucket/xz223/dlenv/PPG/weights/ms1mv2_model_TransFace_S.pt'
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # DFA specific parameters
    lambda_fair: float = 0.01
    mask_prob: float = 0.5
    adapter_scale: float = 1.0
    face_drop_prob: float = 0.1
    face_shuffle_start: float = 0.2
    face_shuffle_end: float = 0.6
    warmup_steps: int = 1000
    
    # Prompt complexity curriculum
    use_prompt_curriculum: bool = True
    prompt_curriculum_warmup: int = 2000
    prompt_curriculum_stages: List[int] = field(default_factory=lambda: [2000, 10000, 20000])
    
    # Optimization
    mixed_precision: bool = True
    num_workers: int = 8
    save_every: int = 5000
    log_every: int = 100
    
    # Resume training
    resume_from: Optional[str] = None
    auto_resume: bool = True


class PromptComplexityCurriculum:
    """Curriculum learning scheduler for prompt complexity"""
    
    def __init__(self, warmup_steps=2000, stage_steps=[2000, 10000, 20000]):
        self.warmup_steps = warmup_steps
        self.stage_steps = stage_steps
        self.current_step = 0
        
        # Progressive complexity weights
        self.stage_weights = [
            {'simple': 1.0, 'medium': 0.0, 'complex': 0.0},  # Warmup: simple only
            {'simple': 0.6, 'medium': 0.4, 'complex': 0.0},  # Stage 1: introduce medium
            {'simple': 0.3, 'medium': 0.5, 'complex': 0.2},  # Stage 2: introduce complex
            {'simple': 0.2, 'medium': 0.4, 'complex': 0.4},  # Stage 3: balanced
            {'simple': 0.1, 'medium': 0.4, 'complex': 0.5},  # Final: focus on complex
        ]
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current complexity weights based on training progress"""
        if self.current_step < self.warmup_steps:
            return self.stage_weights[0]
        
        stage = 0
        for i, threshold in enumerate(self.stage_steps):
            if self.current_step >= threshold:
                stage = i + 1
        
        stage = min(stage, len(self.stage_weights) - 1)
        return self.stage_weights[stage]
    
    def get_stage_name(self) -> str:
        """Get current stage name"""
        if self.current_step < self.warmup_steps:
            return "warmup"
        
        for i, threshold in enumerate(self.stage_steps):
            if self.current_step < threshold:
                return f"stage_{i}"
        
        return "final"
    
    def step(self):
        self.current_step += 1


class FaceConditioningScheduler:
    """Scheduler for face conditioning (drop/shuffle)"""
    
    def __init__(self, total_steps, warmup_steps, drop_prob, shuffle_start, shuffle_end):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.drop_prob = drop_prob
        self.shuffle_start = shuffle_start
        self.shuffle_end = shuffle_end
        self.current_step = 0
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get current drop/shuffle probabilities"""
        if self.current_step < self.warmup_steps:
            return {'drop': 0.0, 'shuffle': 0.0, 'phase': 'warmup'}
        
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        shuffle_prob = self.shuffle_start + (self.shuffle_end - self.shuffle_start) * progress
        
        return {
            'drop': self.drop_prob,
            'shuffle': shuffle_prob,
            'phase': 'training'
        }
    
    def step(self):
        self.current_step += 1


class FACTDataset(Dataset):
    """Dataset with proper complexity loading from processed data"""
    
    def __init__(self, data_root, prompt_curriculum=None):
        self.data_root = Path(data_root)
        self.prompt_curriculum = prompt_curriculum
        
        # Load metadata
        with open(self.data_root / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Load prompts with complexity labels
        with open(self.data_root / 'prompts.json') as f:
            self.prompts = json.load(f)
        
        # Try loading enhanced prompts for variants
        self.enhanced_prompts = {}
        enhanced_path = self.data_root / 'enhanced_prompts.json'
        if enhanced_path.exists():
            with open(enhanced_path) as f:
                self.enhanced_prompts = json.load(f)
        
        self.identity_mapping = metadata.get('identity_mapping', {})
        self.samples = []
        self.samples_by_complexity = {'simple': [], 'medium': [], 'complex': []}
        
        # Build sample list with actual complexity from data processing
        for identity_id, filenames in self.identity_mapping.items():
            for filename in filenames:
                # Get complexity from processed data
                complexity = self.prompts[filename].get('complexity', 'simple')
                prompt = self.prompts[filename].get('prompt', 'a portrait')
                negative_prompt = self.prompts[filename].get('negative_prompt', 'low quality')
                
                # Get variants if available
                variants = []
                if filename in self.enhanced_prompts:
                    variants_dict = self.enhanced_prompts[filename].get('variants', {})
                    if isinstance(variants_dict, dict):
                        variants = list(variants_dict.values())
                    elif isinstance(variants_dict, list):
                        variants = variants_dict
                
                sample = {
                    'filename': filename,
                    'identity': identity_id,
                    'complexity': complexity,
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'variants': variants
                }
                
                self.samples.append(sample)
                self.samples_by_complexity[complexity].append(sample)
        
        # Data transforms
        from torchvision import transforms
        self.face_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Log statistics
        total = len(self.samples)
        logger.info(f"Dataset loaded: {total} samples")
        for complexity in ['simple', 'medium', 'complex']:
            count = len(self.samples_by_complexity[complexity])
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {complexity}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Use curriculum learning to select sample
        if self.prompt_curriculum:
            weights = self.prompt_curriculum.get_current_weights()
            complexities = ['simple', 'medium', 'complex']
            probs = [weights[c] for c in complexities]
            
            # Normalize probabilities
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p/total_prob for p in probs]
            else:
                probs = [1/3, 1/3, 1/3]
            
            # Select complexity based on curriculum
            chosen_complexity = np.random.choice(complexities, p=probs)
            
            # Select sample from chosen complexity
            if self.samples_by_complexity[chosen_complexity]:
                sample = random.choice(self.samples_by_complexity[chosen_complexity])
            else:
                # Fallback to any available sample
                sample = self.samples[idx]
        else:
            sample = self.samples[idx]
        
        # Load images
        image = Image.open(self.data_root / 'images' / sample['filename']).convert('RGB')
        face = Image.open(self.data_root / 'faces' / sample['filename']).convert('RGB')
        
        # Load mask
        mask_path = self.data_root / 'masks' / sample['filename'].replace('.jpg', '.png')
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L')) / 255.0
        else:
            mask = np.ones((512, 512))
        mask_tensor = torch.from_numpy(mask).float()
        
        # Select prompt (potentially use variant)
        prompt = sample['prompt']
        if sample['variants'] and random.random() < 0.3:  # 30% chance to use variant
            prompt = random.choice(sample['variants'])
        
        # Get shuffle face for same identity
        same_identity_files = self.identity_mapping.get(sample['identity'], [])
        if len(same_identity_files) > 1:
            other_files = [f for f in same_identity_files if f != sample['filename']]
            if other_files:
                shuffle_file = random.choice(other_files)
                shuffle_face = Image.open(self.data_root / 'faces' / shuffle_file).convert('RGB')
                shuffle_face_tensor = self.face_transform(shuffle_face)
            else:
                shuffle_face_tensor = self.face_transform(face)
        else:
            shuffle_face_tensor = self.face_transform(face)
        
        return {
            'image': self.image_transform(image),
            'face': self.face_transform(face),
            'shuffle_face': shuffle_face_tensor,
            'mask': mask_tensor,
            'prompt': prompt,
            'negative_prompt': sample['negative_prompt'],
            'identity': sample['identity'],
            'complexity': sample['complexity']
        }


class FaceAttnProcessor(nn.Module):
    """Face attention processor compatible with v1 inference"""
    
    def __init__(self, hidden_size, cross_attention_dim=768, scale=1.0, ln=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.ln = ln
        self.num_ims = 1
        
        # Self-attention and feedforward
        self.attn = SelfAttention(hidden_size, hidden_size)
        self.ff = FeedForward(hidden_size)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Gating parameters
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Store residual
        residual0 = hidden_states
        
        # Handle 4D input
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # Extract face embeddings from encoder hidden states
        if encoder_hidden_states is not None:
            end_pos = encoder_hidden_states.shape[1] - 16 * self.num_ims
            text_hidden_states = encoder_hidden_states[:, :end_pos, :]
            face_hidden_states = encoder_hidden_states[:, end_pos:, :]
            
            if hasattr(attn, 'norm_cross') and attn.norm_cross:
                text_hidden_states = attn.norm_encoder_hidden_states(text_hidden_states)
        else:
            text_hidden_states = encoder_hidden_states
            face_hidden_states = None
        
        # Apply face conditioning with gating
        if face_hidden_states is not None:
            N_visual = hidden_states.shape[1]
            
            # Self-attention with face features
            hidden_states_ln = self.norm1(hidden_states)
            face_hidden_states_ln = self.norm1(face_hidden_states)
            combined = torch.cat([hidden_states_ln, face_hidden_states_ln], dim=1)
            
            attn_out = self.attn(combined, attn)[:, :N_visual, :]
            hidden_states = hidden_states + self.scale * torch.tanh(self.alpha_attn) * attn_out
            
            # Feedforward with gating
            hidden_states_ln = self.norm2(hidden_states)
            ff_out = self.ff(hidden_states_ln)
            hidden_states = hidden_states + self.scale * torch.tanh(self.alpha_dense) * ff_out
        
        # Restore shape for cross-attention
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        # Standard cross-attention with text
        residual = hidden_states
        
        if self.ln is not None:
            hidden_states = self.ln(hidden_states)
        
        if hasattr(attn, 'spatial_norm') and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # Cross-attention computation
        query = attn.to_q(hidden_states)
        key = attn.to_k(text_hidden_states if text_hidden_states is not None else hidden_states)
        value = attn.to_v(text_hidden_states if text_hidden_states is not None else hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        # Apply residual connections
        if hasattr(attn, 'residual_connection') and attn.residual_connection:
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states + residual - residual0
        
        if hasattr(attn, 'rescale_output_factor'):
            hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class SelfAttention(nn.Module):
    """Self-attention module"""
    
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
        
        # Use attention processor's methods if available
        if hasattr(attn, 'head_to_batch_dim'):
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * (q.shape[-1] ** -0.5)
            attn_weights = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn_weights, v)
            
            out = attn.batch_to_head_dim(out)
        else:
            # Fallback implementation
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * (q.shape[-1] ** -0.5)
            attn_weights = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn_weights, v)
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feedforward network"""
    
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
    """GELU-gated linear unit"""
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Identity(nn.Module):
    """Identity layer for replacing layer norms"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class FACTTrainer:
    """Main trainer class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = device
        self.global_step = 0
        
        # Statistics tracking
        self.stats = defaultdict(lambda: defaultdict(list))
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.setup_models()
        self.setup_data()
        self.setup_training()
        
        # Auto-resume if enabled
        if config.auto_resume:
            self.auto_resume()
    
    def setup_models(self):
        """Initialize models"""
        logger.info("Loading models...")
        
        from diffusers import (
            AutoencoderKL,
            UNet2DConditionModel,
            DDPMScheduler
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        
        model_id = self.config.sd_model_id
        
        # Load base models
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32
        ).to(self.device)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        ).to(self.device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        ).to(self.device)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Face components
        self.face_transformer = Face_Transformer(weight=self.config.transface_weight)
        self.face_transformer.eval().to(self.device)
        
        self.face_projector = Face_Prj_Resampler(
            dim=1024, depth=4, dim_head=64, heads=12,
            num_queries=16, embedding_dim=512, output_dim=768, ff_mult=4
        ).to(self.device)
        
        # Inject attention processors
        self.inject_processors()
        
        logger.info("Models loaded successfully")
    
    def inject_processors(self):
        """Inject face attention processors into UNet"""
        self.processors = {}
        layer_norms = {}
        
        unet = self.unet
        
        # Replace layer norms with identity
        for i in range(3):
            for j in range(2):
                if hasattr(unet.down_blocks[i], 'attentions'):
                    ln = unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2
                    layer_norms[f'down_blocks.{i}.attentions.{j}.transformer_blocks.0.attn2.processor'] = ln
                    unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        for i in range(3):
            for j in range(3):
                if hasattr(unet.up_blocks[i+1], 'attentions'):
                    ln = unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2
                    layer_norms[f'up_blocks.{i+1}.attentions.{j}.transformer_blocks.0.attn2.processor'] = ln
                    unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        if hasattr(unet.mid_block, 'attentions'):
            ln = unet.mid_block.attentions[0].transformer_blocks[0].norm2
            layer_norms['mid_block.attentions.0.transformer_blocks.0.attn2.processor'] = ln
            unet.mid_block.attentions[0].transformer_blocks[0].norm2 = Identity()
        
        # Set attention processors
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                from diffusers.models.attention_processor import AttnProcessor
                attn_procs[name] = AttnProcessor()
                continue
            
            if cross_attention_dim is None:
                from diffusers.models.attention_processor import AttnProcessor
                attn_procs[name] = AttnProcessor()
            else:
                processor = FaceAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=self.config.adapter_scale,
                    ln=layer_norms.get(name)
                ).to(self.device)
                attn_procs[name] = processor
                self.processors[name] = processor
        
        unet.set_attn_processor(attn_procs)
        logger.info(f"Injected {len(self.processors)} face attention processors")
    
    def setup_data(self):
        """Setup data loaders"""
        # Initialize prompt curriculum
        self.prompt_curriculum = None
        if self.config.use_prompt_curriculum:
            self.prompt_curriculum = PromptComplexityCurriculum(
                warmup_steps=self.config.prompt_curriculum_warmup,
                stage_steps=self.config.prompt_curriculum_stages
            )
        
        # Create dataset
        dataset = FACTDataset(
            data_root=self.config.data_root,
            prompt_curriculum=self.prompt_curriculum
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"Dataset ready: {len(dataset)} samples")
    
    def setup_training(self):
        """Setup training components"""
        # Collect trainable parameters
        trainable_params = []
        trainable_params.extend(self.face_projector.parameters())
        
        for processor in self.processors.values():
            trainable_params.extend(processor.parameters())
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup schedulers
        total_steps = len(self.dataloader) * self.config.num_epochs
        
        self.face_conditioning_scheduler = FaceConditioningScheduler(
            total_steps=total_steps,
            warmup_steps=self.config.warmup_steps,
            drop_prob=self.config.face_drop_prob,
            shuffle_start=self.config.face_shuffle_start,
            shuffle_end=self.config.face_shuffle_end
        )
        
        logger.info(f"Training setup complete")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def training_step(self, batch):
        """Single training step"""
        images = batch['image'].to(self.device)
        faces = batch['face'].to(self.device)
        shuffle_faces = batch['shuffle_face'].to(self.device)
        masks = batch['mask'].to(self.device)
        prompts = batch['prompt']
        negative_prompts = batch['negative_prompt']
        complexities = batch['complexity']
        
        batch_size = images.shape[0]
        
        # Update complexity statistics
        for complexity in complexities:
            self.stats['complexity'][complexity].append(1)
        
        # Encode images to latents
        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    latents = self.vae.encode(images).latent_dist.sample()
            else:
                latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        # Apply face conditioning with curriculum
        probs = self.face_conditioning_scheduler.get_probabilities()
        
        face_inputs = []
        conditioning_types = []
        for i in range(batch_size):
            rand = random.random()
            if rand < probs['drop']:
                face_inputs.append(torch.zeros_like(faces[i:i+1]))
                conditioning_types.append('drop')
            elif rand < probs['drop'] + probs['shuffle']:
                face_inputs.append(shuffle_faces[i:i+1])
                conditioning_types.append('shuffle')
            else:
                face_inputs.append(faces[i:i+1])
                conditioning_types.append('normal')
        
        face_inputs = torch.cat(face_inputs, dim=0)
        
        # Extract face features
        with torch.no_grad():
            local_features, _ = self.face_transformer(face_inputs)
        
        face_embeddings = self.face_projector(local_features)
        
        # Combine embeddings
        combined_embeddings = torch.cat([text_embeddings, face_embeddings], dim=1)
        
        # UNet forward pass
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=combined_embeddings
                ).sample
        else:
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=combined_embeddings
            ).sample
        
        # Calculate loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        
        # Apply mask with probability
        if random.random() < self.config.mask_prob:
            mask_resized = F.interpolate(
                masks.unsqueeze(1),
                size=(model_pred.shape[-2], model_pred.shape[-1]),
                mode='nearest'
            )
            Mr = mask_resized.repeat(1, model_pred.shape[1], 1, 1)
        else:
            Mr = torch.ones_like(model_pred)
        
        loss = F.mse_loss(model_pred * Mr, target * Mr)
        
        # Update statistics
        self.stats['loss']['total'].append(loss.item())
        for cond_type in ['drop', 'shuffle', 'normal']:
            self.stats['conditioning'][cond_type].append(conditioning_types.count(cond_type))
        
        return loss
    
    def save_checkpoint(self, step):
        """Save checkpoint in v1-compatible format"""
        state_dict = {}
        
        # Save face projector with v1 naming
        for name, param in self.face_projector.state_dict().items():
            state_dict[f'local_fac_prj.{name}'] = param
        
        # Save processor weights
        for proc_name, processor in self.processors.items():
            for param_name, param in processor.state_dict().items():
                state_dict[f'{proc_name}.{param_name}'] = param
        
        # Save with proper naming
        checkpoint_name = f'adapter_dfa_{step:08d}.ckpt'
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        
        torch.save(state_dict, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def auto_resume(self):
        """Auto-resume from latest checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        checkpoints = list(checkpoint_dir.glob('adapter_dfa_*.ckpt'))
        if not checkpoints:
            return
        
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        logger.info(f"Auto-resuming from: {latest}")
        
        # Load checkpoint
        state_dict = torch.load(latest, map_location=self.device)
        
        # Load face projector
        projector_state = {}
        for key, value in state_dict.items():
            if key.startswith('local_fac_prj.'):
                new_key = key.replace('local_fac_prj.', '')
                projector_state[new_key] = value
        
        self.face_projector.load_state_dict(projector_state, strict=False)
        
        # Load processors
        for proc_name, processor in self.processors.items():
            proc_state = {}
            for key, value in state_dict.items():
                if key.startswith(f'{proc_name}.'):
                    new_key = key.replace(f'{proc_name}.', '')
                    proc_state[new_key] = value
            
            if proc_state:
                processor.load_state_dict(proc_state, strict=False)
        
        # Extract step number
        self.global_step = int(latest.stem.split('_')[-1])
        logger.info(f"Resumed from step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting FACT training...")
        
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            for batch in progress_bar:
                loss = self.training_step(batch)
                
                self.optimizer.zero_grad()
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.config.gradient_clip
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                
                # Update schedulers
                self.face_conditioning_scheduler.step()
                if self.prompt_curriculum:
                    self.prompt_curriculum.step()
                
                self.global_step += 1
                epoch_losses.append(loss.item())
                
                # Update progress bar
                postfix = {
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                }
                
                if self.prompt_curriculum:
                    postfix['stage'] = self.prompt_curriculum.get_stage_name()
                
                progress_bar.set_postfix(postfix)
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(self.global_step)
                
                # Log statistics
                if self.global_step % self.config.log_every == 0:
                    self.log_statistics()
                
                # Clear cache periodically
                if self.global_step % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Final save
        self.save_checkpoint(self.global_step)
        logger.info("Training completed!")
    
    def log_statistics(self):
        """Log training statistics"""
        # Calculate averages
        avg_loss = np.mean(self.stats['loss']['total'][-100:]) if self.stats['loss']['total'] else 0
        
        # Complexity distribution
        complexity_counts = {k: sum(v) for k, v in self.stats['complexity'].items()}
        total_complexity = sum(complexity_counts.values())
        
        if total_complexity > 0:
            logger.info(f"Step {self.global_step} - Loss: {avg_loss:.4f}")
            logger.info("Complexity distribution (last 100 steps):")
            for complexity in ['simple', 'medium', 'complex']:
                count = complexity_counts.get(complexity, 0)
                percentage = (count / total_complexity) * 100
                logger.info(f"  {complexity}: {percentage:.1f}%")
        
        # Clear old statistics
        if len(self.stats['loss']['total']) > 1000:
            for key in self.stats['loss']:
                self.stats['loss'][key] = self.stats['loss'][key][-500:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=f'{TMP_BASE}/outputs')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_every=args.save_every,
        resume_from=args.resume
    )
    
    trainer = FACTTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()