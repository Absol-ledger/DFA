
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """Enhanced MLP with improved activation and dropout."""
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        act_layer: Callable = nn.GELU,  # Changed from ReLU6 to GELU
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VITBatchNorm(nn.Module):
    """Batch normalization for ViT with proper handling of different input shapes."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, features) or (batch_size, features)
        if x.dim() == 3:
            # Reshape for BatchNorm1d: (batch_size, features, seq_len)
            batch_size, seq_len, features = x.shape
            x = x.transpose(1, 2)  # (batch_size, features, seq_len)
            x = self.bn(x)
            x = x.transpose(1, 2)  # back to (batch_size, seq_len, features)
        else:
            x = self.bn(x)
        return x


class Attention(nn.Module):
    """Enhanced multi-head attention with improved numerical stability."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, embed_dim = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(
            batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash_attention:
            # Use PyTorch 2.0 flash attention for better memory efficiency
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    x = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                        scale=self.scale
                    )
            except:
                # Fallback to traditional attention
                x = self._traditional_attention(q, k, v)
        else:
            # Traditional attention computation with mixed precision
            x = self._traditional_attention(q, k, v)
        
        # Reshape output
        x = x.transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def _traditional_attention(self, q, k, v):
        """Traditional attention computation"""
        with torch.cuda.amp.autocast(enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        return x


class Block(nn.Module):
    """Enhanced transformer block with improved stability and efficiency."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patches: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: Callable = nn.GELU,
        norm_layer: str = "ln", 
        patch_n: int = 144,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Normalization layers
        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")

        # Attention and MLP
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with enhanced flexibility"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with Face Recognition optimizations"""
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None, 
        representation_size=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=None, 
        mask_ratio=0.0,
        use_checkpoint=False
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            mask_ratio: ratio of patches to mask during training
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mask_ratio = mask_ratio
        norm_layer = norm_layer or nn.LayerNorm

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                num_patches=num_patches + 1,
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer="ln",
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)
        ])
        
        # Final norm
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if mask_ratio == 0:
            return x, None, None
            
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply masking if enabled during training
        if self.training and self.mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        x = self.pos_drop(x)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        """
        Forward pass that returns both local and global features
        Compatible with Face Adapter requirements
        """
        x = self.forward_features(x)
        
        # Extract features
        cls_token = x[:, 0]  # Global feature from cls token
        patch_tokens = x[:, 1:]  # Local features from patches
        
        # Global feature processing
        global_feature = self.pre_logits(cls_token)
        
        # For Face Adapter compatibility:
        # local_features: (batch_size, num_patches, embed_dim)
        # global_features: (batch_size, 1, embed_dim) or (batch_size, embed_dim)
        local_features = patch_tokens  # (B, 144, 512)
        global_features = global_feature.unsqueeze(1)  # (B, 1, 512)
        
        return local_features, global_features


def vit_tiny_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vit_small_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vit_large_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model

