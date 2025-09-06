# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class EnhancedFeedForward(nn.Module):
    """Enhanced feedforward network with improved gating and dropout."""
    
    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        mult: int = 4, 
        glu: bool = False, 
        dropout: float = 0.
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedSelfAttention(nn.Module):
    """Enhanced self-attention with better numerical stability and efficiency."""
    
    def __init__(
        self, 
        query_dim: int, 
        inner_dim: int, 
        dropout: float = 0.,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.query_dim = query_dim
        self.inner_dim = inner_dim
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_processor) -> torch.Tensor:
        """Forward pass with attention processor for head dimension handling."""
        B, N, _ = x.shape
        H = attn_processor.heads
        C = self.inner_dim // H
        
        q = self.to_q(x)
        k = self.to_k(x) 
        v = self.to_v(x)
        
        if self.use_flash_attention:
            # Reshape for flash attention: (B, H, N, C)
            q = q.view(B, N, H, C).transpose(1, 2)
            k = k.view(B, N, H, C).transpose(1, 2)
            v = v.view(B, N, H, C).transpose(1, 2)
            
            # Use flash attention
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,  # Dropout handled separately
                    scale=C ** -0.5
                )
            
            # Reshape back: (B, N, H*C)
            out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        else:
            # Traditional attention computation
            scale = C ** -0.5
            
            q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B*H, N, C)
            k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B*H, N, C)
            v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B*H, N, C)

            sim = torch.einsum('b i c, b j c -> b i j', q, k) * scale
            attn = sim.softmax(dim=-1)
            
            out = torch.einsum('b i j, b j c -> b i c', attn, v)
            out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, self.inner_dim)

        return self.to_out(out)


class AdaptiveGating(nn.Module):
    """Adaptive gating mechanism for controlling feature injection strength."""
    
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Linear(hidden_size // 4, hidden_size // 4),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Global average pooling across sequence dimension
        pooled = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        gate = self.gate_network(pooled)  # (B, 1, 1)
        return gate


class BaseAttnProcessor(nn.Module):
    """Base attention processor with common functionality."""
    
    def __init__(self, hidden_size: Optional[int] = None, cross_attention_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size

    def _prepare_hidden_states(self, hidden_states: torch.Tensor, input_ndim: int) -> Tuple[torch.Tensor, int, int, int, int]:
        """Prepare hidden states for attention computation."""
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = hidden_states.shape[0], None, None, None
            
        return hidden_states, batch_size, channel, height, width

    def _restore_hidden_states(self, hidden_states: torch.Tensor, input_ndim: int, 
                              batch_size: int, channel: int, height: int, width: int) -> torch.Tensor:
        """Restore hidden states to original format."""
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        return hidden_states


class AttnProcessor(BaseAttnProcessor):
    """Standard attention processor for PyTorch < 2.0."""
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnProcessor2_0(BaseAttnProcessor):
    """Optimized attention processor for PyTorch 2.0+ with flash attention."""
    
    def __init__(self, hidden_size: Optional[int] = None, cross_attention_dim: Optional[int] = None):
        super().__init__(hidden_size, cross_attention_dim)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Use flash attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attention_mask, 
                dropout_p=0.0, 
                is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class FaceAttnProcessor(BaseAttnProcessor):
    """Enhanced face attention processor with improved gating and adaptive mechanisms."""
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        ln: Optional[nn.Module] = None,
        use_adaptive_gating: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(hidden_size, cross_attention_dim)
        
        self.scale = scale
        self.ln = ln
        self.num_ims = 1
        self.use_adaptive_gating = use_adaptive_gating
        
        # Feature projection for face embeddings
        self.face_projection = nn.Sequential(
            nn.Linear(self.cross_attention_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Enhanced self-attention and feedforward
        self.self_attn = EnhancedSelfAttention(
            query_dim=hidden_size, 
            inner_dim=hidden_size,
            dropout=dropout
        )
        self.feedforward = EnhancedFeedForward(
            dim=hidden_size, 
            glu=True, 
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Adaptive gating mechanism
        if use_adaptive_gating:
            self.adaptive_gate = AdaptiveGating(hidden_size)
        
        # Learnable gating parameters with better initialization
        self.register_parameter('alpha_attn', nn.Parameter(torch.zeros(1)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.zeros(1)))
        
        # Initialize gating parameters
        nn.init.constant_(self.alpha_attn, 0.1)
        nn.init.constant_(self.alpha_dense, 0.1)

    def _extract_face_features(self, encoder_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract face features from encoder hidden states."""
        end_pos = encoder_hidden_states.shape[1] - 16 * self.num_ims
        text_features = encoder_hidden_states[:, :end_pos, :]
        face_features = encoder_hidden_states[:, end_pos:, :]
        return text_features, face_features

    def _apply_gated_self_attention(self, hidden_states: torch.Tensor, face_features: torch.Tensor, attn) -> torch.Tensor:
        """Apply gated self-attention with face feature injection."""
        N_visual = hidden_states.shape[1]
        
        # Project face features to match hidden dimension
        face_features_proj = self.face_projection(face_features)
        
        # Concatenate visual and face features for self-attention
        combined_features = torch.cat([hidden_states, face_features_proj], dim=1)
        
        # Apply self-attention
        attn_output = self.self_attn(self.norm1(combined_features), attn)
        
        # Take only the visual part and apply gating
        attn_output = attn_output[:, :N_visual, :]
        
        # Adaptive gating if enabled
        if self.use_adaptive_gating:
            adaptive_weight = self.adaptive_gate(face_features_proj)
            gate_weight = self.scale * torch.tanh(self.alpha_attn) * adaptive_weight
        else:
            gate_weight = self.scale * torch.tanh(self.alpha_attn)
        
        return hidden_states + gate_weight * attn_output

    def _apply_gated_feedforward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply gated feedforward network."""
        ff_output = self.feedforward(self.norm2(hidden_states))
        gate_weight = self.scale * torch.tanh(self.alpha_dense)
        return hidden_states + gate_weight * ff_output

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Store original residual
        residual_original = hidden_states
        
        # Ensure proper rescaling and residual connection settings
        assert attn.rescale_output_factor == 1, "Expected rescale_output_factor to be 1"
        assert not attn.residual_connection, "Expected residual_connection to be False"
        
        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)
        
        # Extract face features if available
        if encoder_hidden_states is not None:
            text_features, face_features = self._extract_face_features(encoder_hidden_states)
            if attn.norm_cross:
                text_features = attn.norm_encoder_hidden_states(text_features)
        else:
            text_features = hidden_states
            face_features = None
        
        # Apply gated self-attention with face feature injection
        if face_features is not None:
            hidden_states = self._apply_gated_self_attention(hidden_states, face_features, attn)
            hidden_states = self._apply_gated_feedforward(hidden_states)
        
        # Restore shape for cross-attention
        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)
        
        # Perform standard cross-attention
        residual = hidden_states
        hidden_states = self.ln(hidden_states) if self.ln is not None else hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)

        batch_size, sequence_length, _ = (
            hidden_states.shape if text_features is None else text_features.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(text_features)
        value = attn.to_v(text_features)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)

        # Final residual connection
        hidden_states = hidden_states + residual
        if not attn.residual_connection:
            hidden_states = hidden_states - residual_original

        return hidden_states


class FaceAttnProcessor2_0(FaceAttnProcessor):
    """Enhanced face attention processor with PyTorch 2.0 optimizations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EnhancedFaceAttnProcessor2_0 requires PyTorch 2.0+")
    
    def _perform_cross_attention_2_0(self, attn, hidden_states, text_features, attention_mask, batch_size):
        """Perform cross-attention with PyTorch 2.0 optimizations."""
        query = attn.to_q(hidden_states)
        key = attn.to_k(text_features)
        value = attn.to_v(text_features)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Use flash attention for cross-attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        return hidden_states.to(query.dtype)


# Control Net Processors
class CNAttnProcessor(BaseAttnProcessor):
    """ControlNet attention processor."""
    
    def __init__(self):
        super().__init__()
        self.num_ims = 1

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)
        
        if encoder_hidden_states is not None:
            end_pos = encoder_hidden_states.shape[1] - 16 * self.num_ims
            encoder_hidden_states = encoder_hidden_states[:, :end_pos, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CNAttnProcessor2_0(CNAttnProcessor):
    """ControlNet attention processor with PyTorch 2.0 optimizations."""
    
    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CNAttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        hidden_states, batch_size, channel, height, width = self._prepare_hidden_states(hidden_states, input_ndim)

        if encoder_hidden_states is not None:
            end_pos = encoder_hidden_states.shape[1] - 16 * self.num_ims
            encoder_hidden_states = encoder_hidden_states[:, :end_pos, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = self._restore_hidden_states(hidden_states, input_ndim, batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

