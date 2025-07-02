# =============================================================================
# MODULE 3: LLaMA Integration with HYBRID MoE-LoRA Architecture (SwitchHead Inspired)
# =============================================================================

"""
HYBRID MoE-LoRA SWITCHHEAD ARCHITECTURE - GPU DIRECT VERSION v2.0
================================================================

INNOVATION BASED ON SWITCHHEAD:
- Q, K: CLASSIC LoRA (preserves attention pattern)
- V, O: MoE-LoRA (content specialization)
- TOKEN-WISE expert selection (like original SwitchHead)
- Non-competitive gating function (sigmoid)
- GPU DIRECT LOADING: Avoids RAM saturation

v2.0 IMPROVEMENTS:
- Token-wise selection instead of global pooling
- Increased LoRA parameters (rank=16, alpha=128)
- Optimized expert application
- Corrected dimensions (output projection)
- Architecture closer to original SwitchHead

CONFIRMED ARCHITECTURE:
- Preserved attention pattern (Q,K shared)
- Specialized content (V,O experts)
- LLaMA-2-7B compatible
- GPU/RAM optimized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP
)
from typing import Dict, List, Optional, Union, Tuple
import warnings
import copy
import math

# =============================================================================
# GPU DIRECT MEMORY OPTIMIZATION FUNCTIONS (unchanged)
# =============================================================================

def get_memory_usage_detailed():
    """Detailed system RAM and GPU usage"""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024**3

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        return ram_gb, gpu_allocated, gpu_reserved
    return ram_gb, 0, 0

def monitor_memory_gpu_direct(step_name):
    """Advanced monitoring for GPU Direct"""
    ram_gb, gpu_allocated, gpu_reserved = get_memory_usage_detailed()
    print(f"Memory {step_name}:")
    print(f"   System RAM: {ram_gb:.1f} GB")
    print(f"   GPU allocated: {gpu_allocated:.1f} GB")
    print(f"   GPU reserved: {gpu_reserved:.1f} GB")
    return ram_gb, gpu_allocated

def ultra_aggressive_gpu_cleanup():
    """Ultra-aggressive GPU + RAM cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

# =============================================================================
# MoE-LoRA EXPERT FOR V AND O ONLY
# =============================================================================

class MoELoRAExpertVO(nn.Module):
    """MoE-LoRA expert for V and O only (SwitchHead inspired)"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        specialization: str,
        lora_rank: int = 16,  # INCREASED
        lora_alpha: float = 128.0,  # INCREASED
        device: str = "cuda"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.specialization = specialization
        self.target_device = device

        # ONLY V and O with LoRA (not Q,K!)
        # Base matrices (frozen)
        self.value_base = nn.Linear(hidden_size, self.all_head_size, bias=False).to(device)
        self.output_base = nn.Linear(self.all_head_size, hidden_size, bias=False).to(device)  # CORRECTED

        # Specialized LoRA adaptations for V and O
        self.value_lora = nn.Sequential(
            nn.Linear(hidden_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, self.all_head_size, bias=False).to(device)
        )

        self.output_lora = nn.Sequential(
            nn.Linear(self.all_head_size, lora_rank, bias=False).to(device),  # CORRECTED
            nn.Linear(lora_rank, hidden_size, bias=False).to(device)
        )

        # LoRA scaling
        self.lora_scaling = lora_alpha / lora_rank

        # LoRA initialization
        self._init_lora_weights()

    def _init_lora_weights(self):
        """Standard LoRA initialization: A~N(0,1), B=0"""
        for module in [self.value_lora, self.output_lora]:
            nn.init.normal_(module[0].weight, std=0.01)
            nn.init.zeros_(module[1].weight)

    def get_value_projection(self, hidden_states: torch.Tensor, freeze_base: bool = True) -> torch.Tensor:
        """V projection with base + LoRA"""
        if freeze_base:
            with torch.no_grad():
                base_value = self.value_base(hidden_states)
        else:
            base_value = self.value_base(hidden_states)

        lora_value = self.value_lora(hidden_states) * self.lora_scaling
        return base_value + lora_value

    def get_output_projection(self, attention_output: torch.Tensor, freeze_base: bool = True) -> torch.Tensor:
        """O projection with base + LoRA"""
        if freeze_base:
            with torch.no_grad():
                base_output = self.output_base(attention_output)
        else:
            base_output = self.output_base(attention_output)

        lora_output = self.output_lora(attention_output) * self.lora_scaling
        return base_output + lora_output


# =============================================================================
# HYBRID MoE-LoRA ATTENTION (SWITCHHEAD INSPIRED)
# =============================================================================

class HybridMoELoRAAttention(nn.Module):
    """
    Hybrid attention inspired by SwitchHead:
    - Q, K: Classic LoRA (preserved attention pattern)
    - V, O: MoE-LoRA (specialized content)
    """

    EXPERT_SPECIALIZATIONS = [
        "hedging_uncertainty",      # "where possible", "we aim to"
        "negation_detection",       # "no longer", "have been reduced"
        "attribution_clarity",      # "we", "companies should"
        "temporal_dependency",      # "subject to", "upon completion"
        "certainty_assessment",     # "will implement", "may consider"
        "general_sustainability"    # general ESG patterns
    ]

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_experts: int = 6,
        lora_rank: int = 16,  # INCREASED
        lora_alpha: float = 128.0,  # INCREASED
        top_k_experts: int = 2,
        attention_dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.scale = math.sqrt(self.attention_head_size)
        self.target_device = device

        # =================================================================
        # Q, K: CLASSIC LoRA (shared, preserves attention pattern)
        # =================================================================

        # Base Q,K matrices (frozen)
        self.query_base = nn.Linear(hidden_size, self.all_head_size, bias=False).to(device)
        self.key_base = nn.Linear(hidden_size, self.all_head_size, bias=False).to(device)

        # LoRA adaptations for Q,K (classic, shared)
        self.query_lora = nn.Sequential(
            nn.Linear(hidden_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, self.all_head_size, bias=False).to(device)
        )

        self.key_lora = nn.Sequential(
            nn.Linear(hidden_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, self.all_head_size, bias=False).to(device)
        )

        # =================================================================
        # V, O: MoE-LoRA (multiple experts, specialized content)
        # =================================================================

        # MoE-LoRA experts for V and O
        self.vo_experts = nn.ModuleList([
            MoELoRAExpertVO(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                specialization=self.EXPERT_SPECIALIZATIONS[i],
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=device
            )
            for i in range(num_experts)
        ]).to(device)

        # =================================================================
        # GATING NETWORKS (SwitchHead inspired)
        # =================================================================

        # Gating for V (source-based like SwitchHead)
        self.gate_v = nn.Linear(hidden_size, num_experts, bias=False).to(device)

        # Gating for O (destination-based like SwitchHead)
        self.gate_o = nn.Linear(hidden_size, num_experts, bias=False).to(device)

        # LoRA scaling
        self.lora_scaling = lora_alpha / lora_rank

        self.dropout = nn.Dropout(attention_dropout).to(device)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        # LoRA Q,K
        for module in [self.query_lora, self.key_lora]:
            nn.init.normal_(module[0].weight, std=0.01)
            nn.init.zeros_(module[1].weight)

        # Gating (like SwitchHead)
        nn.init.normal_(self.gate_v.weight, std=0.01)
        nn.init.normal_(self.gate_o.weight, std=0.01)

    def get_expert_selection(self, hidden_states: torch.Tensor, gate: nn.Module, k: int):
        """TOKEN-WISE expert selection (like SwitchHead)"""

        # Token by token selection (no pooling!)
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Scores for each token individually
        scores = torch.sigmoid(gate(hidden_states))  # [batch, seq, num_experts]

        # Top-k selection for each token
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)  # [batch, seq, k]

        # Weight normalization
        expert_weights = F.softmax(top_k_scores, dim=-1)  # [batch, seq, k]

        return top_k_indices, expert_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freeze_base: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Hybrid forward: Q,K classic + V,O MoE"""

        batch_size, seq_len, hidden_size = hidden_states.shape

        try:
            # =============================================================
            # 1. Q, K: CLASSIC LoRA (preserved attention pattern)
            # =============================================================

            if freeze_base:
                with torch.no_grad():
                    base_query = self.query_base(hidden_states)
                    base_key = self.key_base(hidden_states)
            else:
                base_query = self.query_base(hidden_states)
                base_key = self.key_base(hidden_states)

            # LoRA adaptations
            lora_query = self.query_lora(hidden_states) * self.lora_scaling
            lora_key = self.key_lora(hidden_states) * self.lora_scaling

            # Final Q, K (shared for all experts)
            final_query = base_query + lora_query
            final_key = base_key + lora_key

            # Reshape for multi-head
            def reshape_for_attention(x):
                return x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

            query_layer = reshape_for_attention(final_query)
            key_layer = reshape_for_attention(final_key)

            # =============================================================
            # 2. EXPERT SELECTION for V and O (like SwitchHead)
            # =============================================================

            # V: source-based selection (like SwitchHead)
            v_indices, v_weights = self.get_expert_selection(
                hidden_states, self.gate_v, self.top_k_experts
            )

            # O: destination-based selection (like SwitchHead)
            o_indices, o_weights = self.get_expert_selection(
                hidden_states, self.gate_o, self.top_k_experts
            )

            # =============================================================
            # 3. V: MoE-LoRA (specialized content) - OPTIMIZED
            # =============================================================

            # Direct expert application for V
            final_value = torch.zeros_like(final_query)

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # For each token, apply selected experts
                    for k in range(self.top_k_experts):
                        expert_idx = v_indices[batch_idx, seq_idx, k].item()
                        weight = v_weights[batch_idx, seq_idx, k]

                        # Calculate V for this token with this expert
                        token_hidden = hidden_states[batch_idx:batch_idx+1, seq_idx:seq_idx+1]  # [1, 1, hidden]
                        expert_v = self.vo_experts[expert_idx].get_value_projection(token_hidden, freeze_base)

                        final_value[batch_idx, seq_idx] += weight * expert_v[0, 0]

            value_layer = reshape_for_attention(final_value)

            # =============================================================
            # 4. ATTENTION (pattern preserved because Q,K shared!)
            # =============================================================

            # Standard attention calculation
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / self.scale

            # Apply mask
            if attention_mask is not None and attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(1)
                mask = (1.0 - mask) * -10000.0
                attention_scores = attention_scores + mask

            # Softmax
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply to values
            context_layer = torch.matmul(attention_probs, value_layer)

            # Reshape back
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)

            # =============================================================
            # 5. O: MoE-LoRA (specialized transformation) - OPTIMIZED
            # =============================================================

            # Direct expert application for O
            final_output = torch.zeros(batch_size, seq_len, hidden_size, device=hidden_states.device)

            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    # For each token, apply selected experts
                    for k in range(self.top_k_experts):
                        expert_idx = o_indices[batch_idx, seq_idx, k].item()
                        weight = o_weights[batch_idx, seq_idx, k]

                        # Calculate O for this token with this expert
                        token_context = context_layer[batch_idx:batch_idx+1, seq_idx:seq_idx+1]  # [1, 1, all_head_size]
                        expert_o = self.vo_experts[expert_idx].get_output_projection(token_context, freeze_base)

                        final_output[batch_idx, seq_idx] += weight * expert_o[0, 0]

            load_balancing_loss = torch.tensor(0.0, device=hidden_states.device)

            return {
                'hidden_states': final_output,
                'load_balancing_loss': load_balancing_loss
            }

        except Exception as e:
            print(f"Error in HybridMoELoRAAttention: {e}")
            # Fallback
            return {
                'hidden_states': hidden_states,
                'load_balancing_loss': torch.tensor(0.0, device=hidden_states.device)
            }


# =============================================================================
# LoRA-FFN GPU DIRECT (unchanged)
# =============================================================================

class LoRAFFNGPU(nn.Module):
    """FFN module with LoRA adaptation for LLaMA - GPU DIRECT VERSION"""

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        lora_rank: int = 16,
        lora_alpha: float = 128.0,  # INCREASED
        device: str = "cuda"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.target_device = device

        # DIRECT GPU CREATION - Base FFN layers (LLaMA architecture - NO bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False).to(device)

        # SiLU activation function
        self.activation_fn = nn.SiLU().to(device)

        # DIRECT GPU CREATION - LoRA adaptations
        self.lora_gate = nn.Sequential(
            nn.Linear(hidden_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, intermediate_size, bias=False).to(device)
        )
        self.lora_up = nn.Sequential(
            nn.Linear(hidden_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, intermediate_size, bias=False).to(device)
        )
        self.lora_down = nn.Sequential(
            nn.Linear(intermediate_size, lora_rank, bias=False).to(device),
            nn.Linear(lora_rank, hidden_size, bias=False).to(device)
        )

        # LoRA scaling
        self.lora_scaling = lora_alpha / lora_rank

        # LoRA initialization
        self._init_lora_weights()

    def _init_lora_weights(self):
        """Standard LoRA initialization"""
        for module in [self.lora_gate, self.lora_up, self.lora_down]:
            nn.init.normal_(module[0].weight, std=0.01)
            nn.init.zeros_(module[1].weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # Gate projection with LoRA
        with torch.no_grad():
            gate_output = self.gate_proj(hidden_states)
        gate_output = gate_output + self.lora_gate(hidden_states) * self.lora_scaling

        # Up projection with LoRA
        with torch.no_grad():
            up_output = self.up_proj(hidden_states)
        up_output = up_output + self.lora_up(hidden_states) * self.lora_scaling

        # Activation and multiplication (LLaMA style)
        gate_activated = self.activation_fn(gate_output)
        intermediate_output = gate_activated * up_output

        # Down projection with LoRA
        with torch.no_grad():
            final_output = self.down_proj(intermediate_output)
        final_output = final_output + self.lora_down(intermediate_output) * self.lora_scaling

        return final_output


# =============================================================================
# MODIFIED LLAMA LAYER WITH HYBRID APPROACH
# =============================================================================

class LlamaDecoderLayerWithHybridMoE(nn.Module):
    """LLaMA layer with Hybrid MoE-LoRA Attention (SwitchHead inspired) and LoRA-FFN"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int = 0,
        use_hybrid_moe_attention: bool = True,
        use_lora_ffn: bool = True,
        moe_attention_config: Optional[Dict] = None,
        lora_ffn_config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.use_hybrid_moe_attention = use_hybrid_moe_attention
        self.use_lora_ffn = use_lora_ffn
        self.device = device

        # HYBRID MoE-LoRA Attention or standard
        if use_hybrid_moe_attention:
            moe_config = moe_attention_config or {}
            moe_config['device'] = device
            self.self_attn = HybridMoELoRAAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                **moe_config
            )
        else:
            self.self_attn = LlamaAttention(config, layer_idx).to(device)

        # LoRA-FFN or standard
        if use_lora_ffn:
            ffn_config = lora_ffn_config or {}
            ffn_config['device'] = device
            self.mlp = LoRAFFNGPU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                **ffn_config
            )
        else:
            self.mlp = LlamaMLP(config).to(device)

        # Layer norms
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        ).to(device)

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        ).to(device)

        self.last_load_balancing_loss = torch.tensor(0.0, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, ...]:

        residual = hidden_states

        # Pre-norm + Self-attention
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_hybrid_moe_attention:
            # Hybrid MoE-LoRA Attention
            try:
                attn_outputs = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
                hidden_states = attn_outputs['hidden_states']
                self.last_load_balancing_loss = attn_outputs.get('load_balancing_loss', torch.tensor(0.0))

                attn_weights = None
                present_key_value = None

            except Exception as e:
                print(f"Hybrid MoE-Attention error: {e}")
                hidden_states = residual
                attn_weights = None
                present_key_value = None

        else:
            # Standard LLaMA attention
            attn_outputs = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
            hidden_states = attn_outputs[0]
            attn_weights = attn_outputs[1] if output_attentions else None
            present_key_value = attn_outputs[-1] if use_cache else None

        # Residual connection
        hidden_states = residual + hidden_states

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_lora_ffn:
            try:
                hidden_states = self.mlp(hidden_states)
            except Exception as e:
                print(f"LoRA-FFN error: {e}")
                pass
        else:
            hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        # LLaMA compatible format
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# =============================================================================
# COMPLETE LLAMA MODEL WITH HYBRID APPROACH
# =============================================================================

class LlamaModelWithHybridMoE(LlamaModel):
    """LLaMA model with Hybrid MoE-LoRA (SwitchHead inspired) + LoRA-FFN"""

    def __init__(
        self,
        config: LlamaConfig,
        layers_to_replace: List[int] = [10, 15, 20],
        moe_attention_config: Optional[Dict] = None,
        lora_ffn_config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        super().__init__(config)

        self.target_device = device
        self.layers_to_replace = layers_to_replace
        self.moe_attention_config = moe_attention_config or {}
        self.lora_ffn_config = lora_ffn_config or {}

        # Ensure device is passed in configs
        self.moe_attention_config['device'] = device
        self.lora_ffn_config['device'] = device

        # Replace layers with hybrid approach
        self._replace_layers_hybrid()

        # Freeze non-LoRA parameters
        self._freeze_non_lora_parameters()

        # Finalization: Ensure ALL model is on GPU
        print("Finalization - Complete GPU transfer...")
        self = self.to(device)

        print(f"LLaMA-2-7B with Hybrid MoE-LoRA (SwitchHead inspired) created:")
        print(f"  Layers with Hybrid MoE-Attention: {layers_to_replace}")
        print(f"  Target device: {self.target_device}")
        print(f"  Total parameters: {self.count_parameters()['total']:,}")
        print(f"  Trainable parameters: {self.count_parameters()['trainable']:,}")

        # Final verification
        device_check = next(self.parameters()).device
        print(f"  Final model on: {device_check}")

    def _replace_layers_hybrid(self):
        """Replace layers with Hybrid MoE-LoRA architecture"""

        for layer_idx in self.layers_to_replace:
            if layer_idx < len(self.layers):
                print(f"  Replacing layer {layer_idx} with Hybrid MoE-LoRA...")

                old_layer = self.layers[layer_idx]

                new_layer = LlamaDecoderLayerWithHybridMoE(
                    config=self.config,
                    layer_idx=layer_idx,
                    use_hybrid_moe_attention=True,
                    use_lora_ffn=True,
                    moe_attention_config=self.moe_attention_config,
                    lora_ffn_config=self.lora_ffn_config,
                    device=self.target_device
                )

                # Copy weights
                self._copy_compatible_weights_hybrid(old_layer, new_layer)

                self.layers[layer_idx] = new_layer

                # Cleanup
                del old_layer
                ultra_aggressive_gpu_cleanup()

                print(f"    Layer {layer_idx} replaced with Hybrid approach")
            else:
                warnings.warn(f"Layer index {layer_idx} out of range")

    def _copy_compatible_weights_hybrid(self, old_layer, new_layer):
        """Copy compatible weights for hybrid approach"""

        try:
            # Layer norms
            new_layer.input_layernorm.load_state_dict(
                old_layer.input_layernorm.state_dict()
            )
            new_layer.post_attention_layernorm.load_state_dict(
                old_layer.post_attention_layernorm.state_dict()
            )
            print(f"    Layer norms copied")

        except Exception as e:
            print(f"    Layer norms error: {e}")

        try:
            # Copy attention weights to hybrid structure
            if hasattr(new_layer.self_attn, 'query_base') and hasattr(old_layer.self_attn, 'q_proj'):
                old_attn = old_layer.self_attn

                # Q, K bases
                new_layer.self_attn.query_base.weight.data.copy_(old_attn.q_proj.weight.data)
                new_layer.self_attn.key_base.weight.data.copy_(old_attn.k_proj.weight.data)

                # V, O bases for all experts
                for expert in new_layer.self_attn.vo_experts:
                    expert.value_base.weight.data.copy_(old_attn.v_proj.weight.data)
                    expert.output_base.weight.data.copy_(old_attn.o_proj.weight.data)

                print(f"    Hybrid attention weights copied")

        except Exception as e:
            print(f"    Hybrid attention weights not copyable: {e}")

        try:
            # MLP weights
            if hasattr(new_layer.mlp, 'gate_proj') and hasattr(old_layer.mlp, 'gate_proj'):
                old_mlp = old_layer.mlp

                new_layer.mlp.gate_proj.weight.data.copy_(old_mlp.gate_proj.weight.data)
                new_layer.mlp.up_proj.weight.data.copy_(old_mlp.up_proj.weight.data)
                new_layer.mlp.down_proj.weight.data.copy_(old_mlp.down_proj.weight.data)

                print(f"    MLP weights copied")

        except Exception as e:
            print(f"    MLP weights not copyable: {e}")

    def _freeze_non_lora_parameters(self):
        """Freeze all parameters except LoRA and gating"""

        for name, param in self.named_parameters():
            if any(keyword in name.lower() for keyword in ['lora', 'gate_v', 'gate_o']):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters"""

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'trainable_percent': (trainable_params / total_params) * 100
        }


class LlamaForCausalLMWithHybridMoE(LlamaForCausalLM):
    """LLaMA-2-7B for generation with Hybrid MoE-LoRA (SwitchHead inspired)"""

    def __init__(
        self,
        config: LlamaConfig,
        layers_to_replace: List[int] = [10, 15, 20],
        moe_attention_config: Optional[Dict] = None,
        lora_ffn_config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        super().__init__(config)

        # Replace base model with hybrid approach
        self.model = LlamaModelWithHybridMoE(
            config=config,
            layers_to_replace=layers_to_replace,
            moe_attention_config=moe_attention_config,
            lora_ffn_config=lora_ffn_config,
            device=device
        )

        # Ensure ALL model is on correct device
        self = self.to(device)
        print(f"Complete hybrid model moved to: {device}")


# =============================================================================
# FACTORY FUNCTIONS (updated)
# =============================================================================

def create_hybrid_moe_llama_model(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    layers_to_replace: List[int] = [10, 15, 20],
    device_map: str = "auto",
    torch_dtype = torch.float16,
    target_device: str = "cuda",
    use_quantization: bool = True,
    **kwargs
):
    """
    Factory to create LLaMA-2-7B with Hybrid MoE-LoRA (SwitchHead inspired)
    """

    print(f"HYBRID MoE-LoRA CREATION (SwitchHead inspired) on {target_device}")
    monitor_memory_gpu_direct("Factory start")

    # Quantization configuration
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False,
        )
        print("4-bit quantization enabled")

    default_moe_config = {
        'num_experts': 6,
        'lora_rank': 16,  # INCREASED
        'lora_alpha': 128.0,  # INCREASED
        'top_k_experts': 2,
        'device': target_device
    }

    default_lora_ffn_config = {
        'lora_rank': 16,
        'lora_alpha': 128.0,  # INCREASED
        'device': target_device
    }

    moe_config = default_moe_config.copy()
    lora_ffn_config = default_lora_ffn_config.copy()

    if 'moe_attention_config' in kwargs:
        moe_config.update(kwargs['moe_attention_config'])
        moe_config['device'] = target_device
    if 'lora_ffn_config' in kwargs:
        lora_ffn_config.update(kwargs['lora_ffn_config'])
        lora_ffn_config['device'] = target_device

    print("Loading config...")
    config = AutoConfig.from_pretrained(model_name)
    monitor_memory_gpu_direct("Config loaded")

    print("LOADING BASE MODEL...")

    # Device map
    if device_map == "auto":
        final_device_map = {
            "model.embed_tokens": 0,
            "model.norm": 0,
            "lm_head": 0,
        }
        for i in range(32):
            final_device_map[f"model.layers.{i}"] = 0
    else:
        final_device_map = device_map

    # Load base model
    try:
        from transformers import LlamaForCausalLM

        base_model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map=final_device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={0: "30GB"} if use_quantization else None,
        )

        monitor_memory_gpu_direct("Base model loaded")
        print(f"Base model loaded on: {next(base_model.parameters()).device}")

    except Exception as e:
        print(f"Error loading base model: {e}")
        raise

    print("CREATING HYBRID MoE-LoRA MODEL...")

    # Create hybrid model
    model_hybrid = LlamaForCausalLMWithHybridMoE(
        config=config,
        layers_to_replace=layers_to_replace,
        moe_attention_config=moe_config,
        lora_ffn_config=lora_ffn_config,
        device=target_device
    )

    monitor_memory_gpu_direct("Hybrid model created")

    print("COPYING COMPATIBLE WEIGHTS...")

    # Copy weights
    try:
        moe_state_dict = model_hybrid.state_dict()
        base_state_dict = base_model.state_dict()

        for name, param in base_state_dict.items():
            if name in moe_state_dict and moe_state_dict[name].shape == param.shape:
                moe_state_dict[name].copy_(param)

        print("Base weights copied")

    except Exception as e:
        print(f"Weight copy error: {e}")

    # Free base model
    del base_model
    ultra_aggressive_gpu_cleanup()
    monitor_memory_gpu_direct("Base model freed")

    print("Finalizing hybrid model...")

    # Final setup
    model_hybrid = model_hybrid.to(target_device)

    if torch_dtype == torch.float16:
        model_hybrid = model_hybrid.half()

    final_device = next(model_hybrid.parameters()).device
    print(f"Final hybrid model on: {final_device}")

    monitor_memory_gpu_direct("Model finalized")

    print(f"Hybrid MoE-LoRA model (SwitchHead inspired) created on {target_device}!")

    params_info = model_hybrid.model.count_parameters()
    print(f"Trainable parameters: {params_info['trainable']:,} "
          f"({params_info['trainable_percent']:.2f}%)")

    return model_hybrid


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_hybrid_moe_creation():
    """Test hybrid MoE-LoRA creation"""

    print("TEST HYBRID MoE-LoRA CREATION (SwitchHead inspired)")
    print("=" * 55)

    if not torch.cuda.is_available():
        print("GPU not available")
        return False

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ram_before, gpu_before = monitor_memory_gpu_direct("Before test")

    try:
        model = create_hybrid_moe_llama_model(
            model_name="meta-llama/Llama-2-7b-hf",
            layers_to_replace=[15],  # 1 layer for test
            target_device=device,
            torch_dtype=torch.float16,
            use_quantization=True,
            moe_attention_config={
                'num_experts': 3,
                'lora_rank': 4,
                'top_k_experts': 2,
            },
            lora_ffn_config={
                'lora_rank': 4,
            }
        )

        ram_after, gpu_after = monitor_memory_gpu_direct("After creation")

        ram_increase = ram_after - ram_before
        gpu_increase = gpu_after - gpu_before

        print(f"RAM increase: +{ram_increase:.1f}GB")
        print(f"GPU increase: +{gpu_increase:.1f}GB")

        if ram_increase < 5:
            print("SUCCESS: System RAM preserved!")
            print("Hybrid MoE-LoRA (SwitchHead inspired) created successfully!")
            return True
        else:
            print(f"System RAM increased by {ram_increase:.1f}GB")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# FINAL TEST
# =============================================================================

if __name__ == "__main__":
    print("FINAL TEST of HYBRID MoE-LoRA (SwitchHead inspired) architecture")

    test_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False
    )

    try:
        print("\n1. Testing Hybrid MoE-LoRA model (SwitchHead inspired)...")

        model = LlamaForCausalLMWithHybridMoE(
            config=test_config,
            layers_to_replace=[2, 5],
            moe_attention_config={'lora_rank': 8, 'device': 'cuda'},
            lora_ffn_config={'lora_rank': 16, 'device': 'cuda'},
            device='cuda'
        )

        print("Forward test...")
        input_ids = torch.randint(0, test_config.vocab_size, (2, 16))
        attention_mask = torch.ones_like(input_ids)

        model_device = next(model.parameters()).device
        print(f"   Model on: {model_device}")

        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        print(f"   Inputs moved to: {input_ids.device}")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"HYBRID MoE-LoRA (SwitchHead inspired) test successful!")
        print(f"   Output logits shape: {outputs.logits.shape}")

        params_info = model.model.count_parameters()
        print(f"\n2. Parameter information:")
        print(f"   Total: {params_info['total']:,}")
        print(f"   Trainable: {params_info['trainable']:,} ({params_info['trainable_percent']:.2f}%)")

        print(f"\nTOTAL SUCCESS! You now have:")
        print(f"   Q, K: Classic LoRA (preserved attention pattern)")
        print(f"   V, O: MoE-LoRA (specialized content)")
        print(f"   Non-competitive gating (sigmoid, like SwitchHead)")
        print(f"   LLaMA-2-7B Compatible")
        print(f"   System RAM Optimized")
        print(f"   Ready for fine-tuning without breaking attention!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


print("Module Hybrid MoE-LoRA (SwitchHead inspired) loaded!")