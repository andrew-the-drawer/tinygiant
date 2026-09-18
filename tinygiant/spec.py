"""Model geometry and conventions, read from GGUF metadata."""

from dataclasses import dataclass

import numpy as np
from gguf import GGUFValueType


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None or not f.types:
        return default
    t = f.types[0]
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")
    if t == GGUFValueType.ARRAY:
        elem = f.types[-1]
        if elem == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    return f.parts[f.data[0]][0].item()


@dataclass
class ModelSpec:
    arch: str
    n_layers: int
    embed_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_experts: int
    n_experts_used: int
    expert_inter: int
    vocab_size: int
    rms_eps: float
    rope_base: float
    rope_yarn_factor: float = 0.0
    rope_orig_ctx: int = 0
    sliding_window: int = 0
    swa_every: int = 0            # layer i uses the window iff swa_every and i % swa_every == 0
    qk_norm: bool = False
    attn_bias: bool = False
    attn_sinks: bool = False
    router_bias: bool = False
    expert_bias: bool = False
    activation: str = "silu"      # "silu" (SwiGLU) or "gptoss" (clamped, alpha=1.702, +1)
    ffn_norm_name: str = "ffn_norm"

    @property
    def gqa_ratio(self):
        return self.n_heads // self.n_kv_heads

    def is_swa_layer(self, i):
        return self.sliding_window > 0 and self.swa_every > 0 and i % self.swa_every == 0

    @classmethod
    def from_gguf(cls, reader):
        arch = _field(reader, "general.architecture")
        g = lambda k, d=None: _field(reader, f"{arch}.{k}", d)
        names = {t.name for t in reader.tensors}
        n_heads = int(g("attention.head_count"))
        embed = int(g("embedding_length"))
        head_dim = int(g("attention.key_length", embed // n_heads))
        vocab = int(reader.tensor_map["token_embd.weight"].shape[1]) if hasattr(reader, "tensor_map") \
            else int([t for t in reader.tensors if t.name == "token_embd.weight"][0].shape[1])
        spec = cls(
            arch=arch,
            n_layers=int(g("block_count")),
            embed_dim=embed,
            n_heads=n_heads,
            n_kv_heads=int(g("attention.head_count_kv", n_heads)),
            head_dim=head_dim,
            n_experts=int(g("expert_count", 0)),
            n_experts_used=int(g("expert_used_count", 0)),
            expert_inter=int(g("expert_feed_forward_length", g("feed_forward_length"))),
            vocab_size=vocab,
            rms_eps=float(g("attention.layer_norm_rms_epsilon", 1e-6)),
            rope_base=float(g("rope.freq_base", 10000.0)),
            rope_yarn_factor=float(g("rope.scaling.factor", 0.0)) if g("rope.scaling.type") == "yarn" else 0.0,
            rope_orig_ctx=int(g("rope.scaling.original_context_length", 0)),
            sliding_window=int(g("attention.sliding_window", 0)),
            qk_norm="blk.0.attn_q_norm.weight" in names,
            attn_bias="blk.0.attn_q.bias" in names,
            attn_sinks="blk.0.attn_sinks.weight" in names,
            router_bias="blk.0.ffn_gate_inp.bias" in names,
            expert_bias="blk.0.ffn_gate_exps.bias" in names,
        )
        if arch == "gpt-oss":
            spec.activation = "gptoss"
            spec.swa_every = 2
        elif spec.sliding_window:
            spec.swa_every = int(g("attention.sliding_window_pattern", 0))
        if "blk.0.ffn_norm.weight" not in names and "blk.0.post_attention_norm.weight" in names:
            spec.ffn_norm_name = "post_attention_norm"
        elif "blk.0.ffn_norm.weight" not in names and "blk.0.attn_post_norm.weight" in names:
            spec.ffn_norm_name = "attn_post_norm"
        return spec

    def rope_tables(self, max_pos):
        """cos/sin tables [max_pos, head_dim/2], with YaRN scaling when configured."""
        d = self.head_dim
        idx = np.arange(0, d, 2, dtype=np.float64)
        inv_freq = 1.0 / (self.rope_base ** (idx / d))
        mscale = 1.0
        if self.rope_yarn_factor and self.rope_orig_ctx:
            factor, orig = self.rope_yarn_factor, self.rope_orig_ctx
            beta_fast, beta_slow = 32.0, 1.0

            def corr_dim(n_rot):
                return d * np.log(orig / (n_rot * 2 * np.pi)) / (2 * np.log(self.rope_base))

            low = max(int(np.floor(corr_dim(beta_fast))), 0)
            high = min(int(np.ceil(corr_dim(beta_slow))), d - 1)
            ramp = (np.arange(d // 2, dtype=np.float64) - low) / max(high - low, 0.001)
            extrap = 1.0 - np.clip(ramp, 0.0, 1.0)
            inv_freq = (inv_freq / factor) * (1 - extrap) + inv_freq * extrap
            mscale = 0.1 * np.log(factor) + 1.0
        t = np.arange(max_pos, dtype=np.float64)
        freqs = np.outer(t, inv_freq)
        cos = (np.cos(freqs) * mscale).astype(np.float32)
        sin = (np.sin(freqs) * mscale).astype(np.float32)
        return np.ascontiguousarray(cos), np.ascontiguousarray(sin)
