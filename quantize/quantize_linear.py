import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
import torch.nn.functional as F
import qmodel


def quantize_linear_module(
    linear_module: nn.Linear,
    quant_config: Dict[str, Any],
    activation_stats: Dict[str, Any],
    channel_type: str = 'default'
) -> 'QuantizedLinear':
    """
    Replace nn.Linear with quantized version
    """
    # Extract configuration
    w_bit = quant_config['bits']
    scale_factor = quant_config['scale']
    group_size = quant_config['group_size']
    
    # Compute channel-wise scales based on activation statistics
    channel_scales = compute_channel_scales(
        linear_module.weight,
        activation_stats,
        channel_type,
        scale_factor
    )
    
    # Create quantized module based on bit width
    if w_bit == 4:
        # Use AWQ-style WQLinear for INT4
        return create_awq_linear(
            linear_module,
            w_bit=4,
            group_size=group_size,
            channel_scales=channel_scales
        )
    elif w_bit in [6, 8]:
        # Use simpler quantization for INT6/INT8
        return create_int8_linear(
            linear_module,
            w_bit=w_bit,
            group_size=group_size,
            channel_scales=channel_scales
        )
    else:
        raise ValueError(f"Unsupported bit width: {w_bit}")
    
def compute_channel_scales(
    weight: torch.Tensor,
    activation_stats: Dict[str, Any],
    channel_type: str,
    base_scale: float
) -> torch.Tensor:
    """
    Compute per-channel scales based on activation statistics
    """
    out_features = weight.shape[0]
    channel_scales = torch.ones(out_features)
    
    # Find salient channels based on activation patterns
    if channel_type in ['query', 'key']:
        # For Q,K projections, protect channels that contribute to attention spikes
        spike_channels = activation_stats.get('spike_channels', [])
        if len(spike_channels) > 0:
            # Scale up salient channels (smaller scale = more protection in AWQ)
            channel_scales[spike_channels] = 1.0 / base_scale
            # Normal channels get standard scaling
            non_spike_channels = [i for i in range(out_features) if i not in spike_channels]
            channel_scales[non_spike_channels] = 1.0
    
    elif channel_type == 'value':
        # V projection can use more aggressive quantization
        channel_scales[:] = 1.0
        
    else:  # output or ffn
        # Use weight magnitude to determine importance
        weight_importance = weight.abs().mean(dim=1)
        top_k = int(0.01 * out_features)  # Top 1%
        important_channels = torch.topk(weight_importance, top_k).indices
        channel_scales[important_channels] = 1.0 / (base_scale * 0.5)
    
    return channel_scales

def create_awq_linear(
    linear_module: nn.Linear,
    w_bit: int,
    group_size: int,
    channel_scales: torch.Tensor
) -> 'WQLinear':
    """
    Create AWQ-style quantized linear layer with debugging
    """
    from qmodel import WQLinear, pack_intweight
    
    device = linear_module.weight.device
    
    # Step 1: Apply channel-wise scaling
    scaled_weight = linear_module.weight.data.clone()
    scaled_weight = scaled_weight * channel_scales.unsqueeze(1)
    
    # Step 2: Compute quantization parameters
    w_quantized, scales, zeros = compute_quantization_params(
        scaled_weight,
        w_bit=w_bit,
        group_size=group_size
    )
    
    print(f"After compute_quantization_params:")
    print(f"  scales dtype: {scales.dtype}, shape: {scales.shape}")
    print(f"  zeros dtype: {zeros.dtype}, shape: {zeros.shape}")
    
    # Convert to FP16
    scales = scales.half()
    zeros = zeros.half() if zeros.dtype.is_floating_point else zeros
    
    # Step 3: Create empty WQLinear module
    q_linear = WQLinear(
        w_bit=w_bit,
        group_size=group_size,
        in_features=linear_module.in_features,
        out_features=linear_module.out_features,
        bias=linear_module.bias is not None,
        dev=device,
        dtype=torch.float16
    )
    
    print(f"After WQLinear creation:")
    print(f"  q_linear.scales dtype: {q_linear.scales.dtype}, shape: {q_linear.scales.shape}")
    
    # Step 4: Pack quantized weights
    w_quantized_int32 = w_quantized.to(torch.int32).contiguous()
    packed_weight = pack_intweight(w_quantized_int32, interleave=4, kstride=64)
    q_linear.qweight.data = packed_weight
    
    # Step 5: Adjust and set scales/zeros
    adjusted_scales = scales / channel_scales.unsqueeze(1).half()
    
    print(f"Before assignment:")
    print(f"  adjusted_scales dtype: {adjusted_scales.dtype}, shape: {adjusted_scales.shape}")
    print(f"  Need to transpose from {adjusted_scales.shape} to {q_linear.scales.shape}")
    
    # AWQ expects transposed scales
    q_linear.scales.data = adjusted_scales.T.contiguous().half()
    
    print(f"After assignment:")
    print(f"  q_linear.scales dtype: {q_linear.scales.dtype}")
    
    # Pre-compute scaled zeros
    scaled_zeros = -(scales * zeros)
    q_linear.scaled_zeros.data = scaled_zeros.T.contiguous().half()
    
    # Step 6: Copy bias if exists
    if linear_module.bias is not None:
        q_linear.bias.data = linear_module.bias.data.clone().half()
    
    # Final check
    print(f"\nFinal WQLinear state:")
    print(f"  scales dtype: {q_linear.scales.dtype}")
    print(f"  qweight dtype: {q_linear.qweight.dtype}")
    if hasattr(q_linear, 'qzeros'):
        print(f"  qzeros dtype: {q_linear.qzeros.dtype}")
    
    return q_linear

def create_int8_linear(
    linear_module: nn.Linear,
    w_bit: int,
    group_size: int,
    channel_scales: torch.Tensor
) -> 'QuantizedLinear':
    """
    Optimized INT6/INT8 quantization without CUDA dependency
    """
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
                
        def forward(self, x):
            # Efficient dequantization and matmul
            # For better performance, implement fused kernel
            weight_fp = self._dequantize_weights()
            return F.linear(x, weight_fp, self.bias)
            
        def _dequantize_weights(self):
            """Optimized dequantization"""
            weight_fp = torch.zeros(
                (self.out_features, self.in_features),
                dtype=torch.float16,
                device=self.qweight.device
            )
            
            # Vectorized dequantization
            for g in range(0, self.in_features, self.group_size):
                end = min(g + self.group_size, self.in_features)
                g_idx = g // self.group_size
                
                weight_fp[:, g:end] = (
                    self.qweight[:, g:end].to(torch.float16) - 
                    self.zeros[:, g_idx:g_idx+1]
                ) * self.scales[:, g_idx:g_idx+1]
                
            return weight_fp
    
    # Create quantized module
    q_linear = QuantizedLinear(
        linear_module.in_features,
        linear_module.out_features,
        linear_module.bias is not None,
        w_bit,
        group_size
    )
    
    # Apply channel scaling
    scaled_weight = linear_module.weight.data * channel_scales.unsqueeze(1)
    
    # Quantize
    w_quantized, scales, zeros = compute_quantization_params(
        scaled_weight,
        w_bit=w_bit,
        group_size=group_size
    )
    
    # Store quantized values
    if w_bit == 8:
        q_linear.qweight.copy_(w_quantized.to(torch.int8))
    else:  # 6-bit
        # Pack 6-bit values (you'd implement 6-bit packing)
        q_linear.qweight.copy_(w_quantized.clamp(0, 63).to(torch.int8))
    
    # Adjust scales for channel scaling
    q_linear.scales.copy_((scales / channel_scales.unsqueeze(1)).to(torch.float16))
    q_linear.zeros.copy_(zeros.to(torch.float16))
    
    if linear_module.bias is not None:
        q_linear.bias.copy_(linear_module.bias)
    
    return q_linear

def compute_quantization_params(
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

def analyze_attention_for_scaling(attention_dist) -> Dict[str, Any]:
    """
    Analyze attention distribution to find important channels
    """
    # Flatten all attention values
    values = attention_dist.flatten()
    values = values[values > 1e-10]
    
    # Find spike range (where most attention mass is)
    hist, bins = np.histogram(np.log10(values), bins=50)
    spike_idx = np.argmax(hist)
    spike_range = (10**bins[spike_idx], 10**bins[spike_idx+1])
    
    # Identify channels that contribute to the spike
    # This is a simplified version - in practice, you'd trace back
    # which channels produce attention values in the spike range
    spike_threshold = spike_range[0]
    spike_mask = attention_dist > spike_threshold
    spike_channels = torch.where(spike_mask.any(dim=-1).any(dim=0))[0].tolist()
    
    return {
        'spike_range': spike_range,
        'spike_channels': spike_channels[:int(0.01 * len(spike_channels))],  # Top 1%
        'mean': values.mean().item(),
        'std': values.std().item(),
        '99th_percentile': np.percentile(values, 99)
    }

def quantize_layer_with_config(layer, config: Dict[str, Any], attention_dist):
    """
    Replace linear projections with quantized versions based on config
    """
    # First, collect activation statistics for this layer
    activation_stats = analyze_attention_for_scaling(attention_dist)

    
    # Quantize attention projections
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        
        # Q projection
        if hasattr(attn, 'q_proj'):
            attn.q_proj = quantize_linear_module(
                attn.q_proj,
                config['attention']['q_proj'],
                activation_stats,
                channel_type='query'
            )
        
        # K projection  
        if hasattr(attn, 'k_proj'):
            attn.k_proj = quantize_linear_module(
                attn.k_proj,
                config['attention']['k_proj'],
                activation_stats,
                channel_type='key'
            )
        
        # V projection
        if hasattr(attn, 'v_proj'):
            attn.v_proj = quantize_linear_module(
                attn.v_proj,
                config['attention']['v_proj'],
                activation_stats,
                channel_type='value'
            )
        
        # Output projection
        if hasattr(attn, 'out_proj'):
            attn.out_proj = quantize_linear_module(
                attn.out_proj,
                config['attention']['out_proj'],
                activation_stats,
                channel_type='output'
            )
    
    # Quantize FFN projections
    '''
    if hasattr(layer, 'mlp') or hasattr(layer, 'ffn'):
        ffn = layer.mlp if hasattr(layer, 'mlp') else layer.ffn
        
        for proj_name in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']:
            if hasattr(ffn, proj_name):
                ffn_config_key = proj_name if proj_name in config['ffn'] else 'gate_proj'
                setattr(ffn, proj_name, quantize_linear_module(
                    getattr(ffn, proj_name),
                    config['ffn'][ffn_config_key],
                    activation_stats,
                    channel_type='ffn'
                ))
    '''

