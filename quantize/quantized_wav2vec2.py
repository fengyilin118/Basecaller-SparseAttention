import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from qmodel import WQLinear, pack_intweight


class QuantizedAttention(nn.Module):
    """
    Self-contained quantized attention module.
    Usage: q_attn = QuantizedAttention.from_float(original_attn, layer_idx, calibration_data)
    """
    def __init__(self, embed_dim=768, num_heads=12, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # These will be populated by from_float()
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        
        # Store quantization config
        self.quant_config = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard attention forward pass"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights
        
    @classmethod
    def from_float(
        cls,
        float_attention: nn.Module,
        layer_idx: int,
        #calibration_data: torch.Tensor,
        device: str = 'cuda'
    ) -> 'QuantizedAttention':
        """
        Create quantized attention from floating point attention module
        
        Args:
            float_attention: Original attention module with nn.Linear projections
            layer_idx: Layer index (0-11) for distribution-aware quantization
            calibration_data: Input data for activation analysis [batch, seq, hidden]
            device: Device for computation
        
        Returns:
            QuantizedAttention module ready for inference
        """
        # Create new quantized attention instance
        q_attn = cls(
            embed_dim=float_attention.embed_dim,
            num_heads=float_attention.num_heads,
            bias=float_attention.q_proj.bias is not None
        )
        
        # Move to device
        float_attention = float_attention.to(device)
        #calibration_data = calibration_data.to(device)
        
        # Analyze attention patterns
        '''
        print(f"Analyzing attention patterns for layer {layer_idx}...")
        with torch.no_grad():
            _, attn_weights = float_attention(calibration_data)
            attn_stats = q_attn._analyze_attention_distribution(attn_weights, layer_idx)
        '''
        
        # Store quantization config
        #q_attn.quant_config = q_attn._determine_quant_config(attn_stats, layer_idx)

        if layer_idx in [0,1,2,3,8,9,10,11]:
            q_attn.quant_config = {
                'distribution_type': 'concentrated',
                'base_scale': 1000,  # Scale 1e-3 to ~1
                'q_config': {'bits': 8, 'group_size': 64},
                'k_config': {'bits': 8, 'group_size': 64},
                'v_config': {'bits': 6, 'group_size': 128},
                'out_config': {'bits': 4, 'group_size': 128},
            }
        else:
            q_attn.quant_config  = {
                'distribution_type': 'medium',
                'base_scale': 100,
                'q_config': {'bits': 6, 'group_size': 96},
                'k_config': {'bits': 6, 'group_size': 96},
                'v_config': {'bits': 4, 'group_size': 128},
                'out_config': {'bits': 4, 'group_size': 128},
            }

        
        # Quantize each projection
        print(f"Quantizing projections with {q_attn.quant_config['distribution_type']} config...")
        
        # Q projection
        q_attn.q_proj = q_attn._quantize_projection(
            float_attention.q_proj,
            #calibration_data,
            #attn_stats,
            proj_type='query',
            config_key='q_config'
        )
        
        # K projection
        q_attn.k_proj = q_attn._quantize_projection(
            float_attention.k_proj,
            #calibration_data,
            #attn_stats,
            proj_type='key',
            config_key='k_config'
        )
        
        # V projection
        q_attn.v_proj = q_attn._quantize_projection(
            float_attention.v_proj,
            #calibration_data,
            #attn_stats,
            proj_type='value',
            config_key='v_config'
        )
        
        # Output projection
        q_attn.out_proj = q_attn._quantize_projection(
            float_attention.out_proj,
            #calibration_data,
            #attn_stats,
            proj_type='output',
            config_key='out_config'
        )
        
        return q_attn
    
    def _quantize_projection(
        self,
        float_proj: nn.Linear,
        #calibration_data: torch.Tensor,
        #attn_stats: dict,
        proj_type: str,
        config_key: str
    ) -> nn.Module:
        """Quantize a single projection module"""
        config = self.quant_config[config_key]
        w_bit = config['bits']
        group_size = config['group_size']
        
        # Identify critical channels and compute scales
        channel_info = {}
        
        channel_info = self._identify_critical_channels(float_proj.weight, proj_type)
        
        channel_scales = channel_info['channel_scales']
        
        
        print(f"  {proj_type}: INT{w_bit}, group_size={group_size}, scales={channel_scales.mean().item():.4f}, "
              #f"spike_ch={len(channel_info['spike_channels'])}, "
              #f"tiny_ch={len(channel_info['tiny_channels'])}"
              )
        
        if w_bit == 4:
            # Use AWQ WQLinear for INT4
            return self._create_wqlinear(float_proj, group_size, channel_scales)
        else:
            # Use custom INT6/INT8 implementation
            return self._create_int8_linear(float_proj, w_bit, group_size, channel_scales)
    
    def _create_wqlinear(
        self,
        float_linear: nn.Linear,
        group_size: int,
        channel_scales: torch.Tensor
    ) -> WQLinear:
        """Create AWQ INT4 quantized linear module"""
        # Scale weights
        scaled_weight = float_linear.weight.data * channel_scales.unsqueeze(1)
        
        # Quantize
        
        w_quantized, scales, zeros = self.compute_quantization_params(
            scaled_weight,
            4,
            group_size
        )
        
        # Adjust scales to account for channel scaling
        adjusted_scales = scales / channel_scales.unsqueeze(1)
        
        # Create temporary module with scaled weights
        temp_linear = nn.Linear(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None
        )
        temp_linear.weight.data = float_linear.weight.data.clone()
        if float_linear.bias is not None:
            temp_linear.bias.data = float_linear.bias.data.clone()
        
        # Create WQLinear
        q_linear = WQLinear.from_linear(
            temp_linear,
            w_bit=4,
            group_size=group_size,
            init_only=False,
            scales=adjusted_scales.half(),
            zeros=zeros
        )
        
        return q_linear
    
    def _create_int8_linear(
        self,
        float_linear: nn.Linear,
        w_bit: int,
        group_size: int,
        channel_scales: torch.Tensor
    ) -> nn.Module:
        """Create INT6/INT8 quantized linear module"""
        
        # Scale weights
        scaled_weight = float_linear.weight.data * channel_scales.unsqueeze(1)
        
        # Create quantized module
        q_linear = QuantizedLinear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            w_bit,
            group_size
        )
        
        # Quantize and store
        q_linear.quantize_and_store(scaled_weight, channel_scales)
        
        if float_linear.bias is not None:
            q_linear.bias.data = float_linear.bias.data.clone()
        
        return q_linear
    
    def _identify_critical_channels(
        self,
        weight: torch.Tensor,
        #calibration_data: torch.Tensor,
        #attn_stats: dict,
        proj_type: str
    ) -> dict[str, torch.Tensor]:
        """Identify channels that need special scaling"""
        out_features, in_features = weight.shape
        '''
        # Compute channel outputs
        with torch.no_grad():
            channel_outputs = calibration_data @ weight.T  # [batch, seq, out_features]
            channel_magnitudes = channel_outputs.abs().mean(dim=(0, 1))  # [out_features]
        '''
        # Initialize scales
        channel_scales = torch.ones(out_features, device=weight.device)
        base_scale = self.quant_config['base_scale']
        
        if proj_type in ['query', 'key']:
            # Find channels producing values near spike
            '''
            spike_loc = attn_stats['spike_location']
            spike_range = (spike_loc / 3, spike_loc * 3)
            
            spike_channels = torch.where(
                (channel_magnitudes > spike_range[0]) & 
                (channel_magnitudes < spike_range[1])
            )[0]
            
            # Find channels producing tiny values
            tiny_threshold = attn_stats['p10']
            tiny_channels = torch.where(channel_magnitudes < tiny_threshold)[0]
            
            # Apply differential scaling
            if len(spike_channels) > 0:
                # Limit to top 10%
                n_spike = min(len(spike_channels), int(0.1 * out_features))
                top_spike = spike_channels[torch.topk(channel_magnitudes[spike_channels], n_spike).indices]
                channel_scales[top_spike] = base_scale * 2
            
            if len(tiny_channels) > 0:
                # Limit to top 5%
                n_tiny = min(len(tiny_channels), int(0.05 * out_features))
                top_tiny = tiny_channels[torch.topk(channel_magnitudes[tiny_channels], n_tiny, largest=False).indices]
                channel_scales[top_tiny] = base_scale * 10
            '''
            # Regular channels
            regular_mask = channel_scales == 1.0
            channel_scales[regular_mask] = base_scale
            
        elif proj_type == 'value':
            # V projection: uniform moderate scaling
            channel_scales[:] = base_scale / 5
            
        else:  # output
            # Scale based on weight magnitude
            if weight.dtype == torch.float16:
                weight_norms = weight.abs().mean(dim=1).float()  # Convert to FP32
                bottom_20_percent = torch.quantile(weight_norms, 0.2)
            else:
                weight_norms = weight.abs().mean(dim=1)
                bottom_20_percent = torch.quantile(weight_norms, 0.2)

            small_weight_channels = weight_norms < bottom_20_percent
            
            channel_scales[:] = base_scale / 10
            channel_scales[small_weight_channels] = base_scale / 5
        
        return {
            'channel_scales': channel_scales,
            #'spike_channels': spike_channels if proj_type in ['query', 'key'] else [],
            #'tiny_channels': tiny_channels if proj_type in ['query', 'key'] else []
        }
    
    def _determine_quant_config(self, attn_stats: dict, layer_idx: int) -> dict:
        """Determine quantization configuration based on attention statistics"""
        if attn_stats['distribution_type'] == 'concentrated':
            # Layers 0-3, 8-11: Need careful quantization
            config = {
                'distribution_type': 'concentrated',
                'base_scale': 1000,  # Scale 1e-3 to ~1
                'q_config': {'bits': 8, 'group_size': 64},
                'k_config': {'bits': 8, 'group_size': 64},
                'v_config': {'bits': 6, 'group_size': 128},
                'out_config': {'bits': 4, 'group_size': 128},
            }
        elif attn_stats['distribution_type'] == 'broad':
            # Layer 7: Can use aggressive quantization
            config = {
                'distribution_type': 'broad',
                'base_scale': 10,
                'q_config': {'bits': 4, 'group_size': 128},
                'k_config': {'bits': 4, 'group_size': 128},
                'v_config': {'bits': 4, 'group_size': 128},
                'out_config': {'bits': 4, 'group_size': 128},
            }
        else:
            # Layers 4-6: Medium approach
            config = {
                'distribution_type': 'medium',
                'base_scale': 100,
                'q_config': {'bits': 6, 'group_size': 96},
                'k_config': {'bits': 6, 'group_size': 96},
                'v_config': {'bits': 4, 'group_size': 128},
                'out_config': {'bits': 4, 'group_size': 128},
            }
        
        return config
    
    def compute_quantization_params(
        self,
        weight: torch.Tensor,
        w_bit: int,
        group_size: int
    ) -> tuple:
        """
        Compute quantization parameters (from AWQ)
        """
        org_shape = weight.shape
    
        # Reshape for group-wise quantization
        if group_size > 0 and org_shape[1] % group_size == 0:
            weight = weight.reshape(-1, group_size)
    
        # Compute scales and zero points
        max_val = weight.amax(dim=1, keepdim=True)
        min_val = weight.amin(dim=1, keepdim=True)
    
        max_int = 2**w_bit - 1
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp(0, max_int)
    
        # Quantize
        w_quantized = torch.clamp(
            torch.round(weight / scales) + zeros,
            0, max_int
        )
    
        # Reshape back
        w_quantized = w_quantized.reshape(org_shape)
        scales = scales.reshape(org_shape[0], -1)
        zeros = zeros.reshape(org_shape[0], -1)
    
        return w_quantized, scales, zeros

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, w_bit, group_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
            
        # Store quantized weights more efficiently
        if w_bit == 8:
            dtype = torch.int8
        else:  # 6-bit stored in int8
            dtype = torch.int8
            
        self.register_buffer('qweight', torch.zeros(
            (out_features, in_features), dtype=dtype
        ))
            
        # Scales and zeros per group
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer('scales', torch.zeros(
            (out_features, num_groups), dtype=torch.float16
        ))
        self.register_buffer('zeros', torch.zeros(
            (out_features, num_groups), dtype=torch.float16
        ))
            
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    def quantize_and_store(self, scaled_weight: torch.Tensor, channel_scales: torch.Tensor):
        """
        Quantize scaled weights and store them along with adjusted scales
        
        Args:
            scaled_weight: Weight matrix already scaled by channel_scales
            channel_scales: Per-channel scaling factors used
        """
        # Reshape for group-wise quantization
        org_shape = scaled_weight.shape
        if self.group_size > 0 and org_shape[1] % self.group_size == 0:
            grouped_weight = scaled_weight.reshape(-1, self.group_size)
        else:
            # Handle non-divisible case
            grouped_weight = scaled_weight.reshape(org_shape[0], -1, self.group_size)
            grouped_weight = grouped_weight.reshape(-1, self.group_size)
        
        # Find min/max per group
        max_vals = grouped_weight.amax(dim=1, keepdim=True)
        min_vals = grouped_weight.amin(dim=1, keepdim=True)
        
        # Compute scales and zeros
        max_int = 2**self.w_bit - 1
        scales = (max_vals - min_vals).clamp(min=1e-6) / max_int
        zeros = (-torch.round(min_vals / scales)).clamp(0, max_int)
        
        # Quantize
        w_quantized = torch.clamp(
            torch.round(grouped_weight / scales) + zeros,
            0, max_int
        )
        
        # Reshape back
        w_quantized = w_quantized.reshape(org_shape)
        scales = scales.reshape(org_shape[0], -1)
        zeros = zeros.reshape(org_shape[0], -1)
        
        # Adjust scales to account for channel scaling
        # This is crucial: during inference, we want to recover original weights
        adjusted_scales = scales / channel_scales.unsqueeze(1)
        
        # Store quantized values
        if self.w_bit == 8:
            self.qweight.copy_(w_quantized.to(torch.int8))
        else:  # 6-bit
            # For 6-bit, ensure values fit in 6 bits
            assert w_quantized.max() <= 63, f"6-bit overflow: max value {w_quantized.max()}"
            self.qweight.copy_(w_quantized.to(torch.int8))
        
        # Store adjusted scales and zeros
        self.scales.copy_(adjusted_scales.to(torch.float16))
        self.zeros.copy_(zeros.to(torch.float16))
                
    def forward(self, x):
        # Efficient dequantization and matmul
        # For better performance, implement fused kernel
        # Ensure dequantized weights are on same device as input


        weight_fp = self._dequantize_weights(
            target_dtype=x.dtype,
            target_device=x.device
        )
        
        # Ensure bias matches input dtype
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype, device=x.device)
        
        return F.linear(x, weight_fp, bias)
            
    def _dequantize_weights(self, target_dtype=torch.float32, target_device=None):
        """Optimized dequantization"""
        if target_device is None:
            target_device = self.qweight.device

        weight_fp = torch.zeros(
            (self.out_features, self.in_features),
            dtype=target_dtype,
            device=target_device
        )
            
        # Convert scales and zeros to match target dtype
        scales = self.scales.to(device=target_device, dtype=target_dtype)
        zeros = self.zeros.to(device=target_device, dtype=target_dtype)
        qweight = self.qweight.to(device=target_device)
        
        # Vectorized dequantization
        for g in range(0, self.in_features, self.group_size):
            end = min(g + self.group_size, self.in_features)
            g_idx = g // self.group_size
            
            # All operations in target dtype
            weight_fp[:, g:end] = (
                qweight[:, g:end].to(target_dtype) - zeros[:, g_idx:g_idx+1]
            ) * scales[:, g_idx:g_idx+1]
        
        return weight_fp

def quantize_wav2vec_attention(
    fine_model: nn.Module,
    layers_to_quantize: list[int] = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    Quantize attention layers in wav2vec model
    
    Args:
        fine_model: Your fine-tuned wav2vec model
        calibration_data: Pre-computed calibration data (optional)
        audio_files: List of audio files for calibration (optional)
        layers_to_quantize: Specific layers to quantize (default: all)
    
    Returns:
        Model with quantized attention layers
    """
    
    # Default to quantizing all layers
    if layers_to_quantize is None:
        layers_to_quantize = list(range(12))
    
    # Store original forward method
    original_forward = fine_model.forward
    
    # Process each layer
    for layer_idx in layers_to_quantize:
        print(f"\n{'='*60}")
        print(f"Quantizing attention layer {layer_idx}")
        print(f"{'='*60}")
        
        # Get the attention module
        original_attn = fine_model.sub_model.encoder.layers[layer_idx].attention
        
        # Check if it's already quantized
        if hasattr(original_attn, 'quant_config'):
            print(f"Layer {layer_idx} already quantized, skipping...")
            continue
        
        try:
            # Create quantized attention
            quantized_attn = QuantizedAttention.from_float(
                original_attn,
                layer_idx=layer_idx,
                device=device
            )
            
            # Replace the attention module
            fine_model.sub_model.encoder.layers[layer_idx].attention = quantized_attn
            
            # Verify the replacement
            print(f"Successfully quantized layer {layer_idx}")
            
        except Exception as e:
            print(f"Error quantizing layer {layer_idx}: {e}")
            print("Keeping original attention module")
    
    return fine_model