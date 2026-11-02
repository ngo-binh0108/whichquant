"""Estimate the memory llama.cpp needs for a GGUF model, and how to place it.

The weights are exact (summed from the tensor table). The KV cache and the
recurrent state follow llama.cpp's own sizing rules (src/llama-kv-cache*.cpp,
src/llama-hparams.cpp). The compute buffer is an estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from whichquant.gguf import GGUFHeader, Tensor

GiB = 1024**3
MiB = 1024**2

# bytes per element of the KV cache types llama.cpp accepts for -ctk / -ctv
CACHE_TYPES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

# Sliding-window layout llama.cpp hardcodes when the GGUF does not carry
# <arch>.attention.sliding_window_pattern: arch -> (pattern, default window).
# With pattern n, every n-th layer uses full attention and the others the window.
SWA_DEFAULTS = {"gemma2": (2, 4096), "gemma3": (6, 0), "cohere2": (4, 0), "gpt-oss": (2, 0), "llama4": (4, 8192)}
# Hybrids with one full-attention layer every n layers and recurrent layers between.
RECURRENT_INTERVAL = {"qwen3next": 4, "qwen35": 4, "qwen35moe": 4}
PURE_RECURRENT = ("mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7")

_BLOCK = re.compile(r"^blk\.(\d+)\.")


def _pad(n: int, to: int = 256) -> int:
    return -(-n // to) * to


def _per_layer(value, n_layer: int) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value[:n_layer]] + [int(value[-1])] * max(0, n_layer - len(value))
    return [int(value)] * n_layer


@dataclass
class Layout:
    """Where the bytes of one model are: per-layer weights and per-token cache."""

    arch: str
    n_layer: int
    n_ctx_train: int
    n_embd: int
    n_vocab: int
    n_ff: int
    layer_bytes: list[int]
    expert_bytes: list[int]  # part of layer_bytes that --n-cpu-moe can keep in RAM
    input_bytes: int  # token embeddings: llama.cpp keeps these in RAM
    output_bytes: int  # output head and norms: on the GPU when every layer is
    kv_full: list[float]  # cache bytes per token per layer (0 for layers without a full-context cache)
    kv_swa: list[float]  # same for sliding-window layers
    n_swa: int
    recurrent_bytes: list[int]  # fixed state per layer for recurrent layers
    n_experts: int = 0
    n_experts_used: int = 0
    notes: list[str] = field(default_factory=list)

    def kv_bytes(self, n_ctx: int, n_ubatch: int = 512, layers: range | None = None) -> float:
        cells = _pad(n_ctx)
        swa_cells = min(cells, _pad(self.n_swa + n_ubatch)) if self.n_swa else cells
        layers = layers if layers is not None else range(self.n_layer)
        return sum(self.kv_full[i] * cells + self.kv_swa[i] * swa_cells + self.recurrent_bytes[i] for i in layers)

    def compute_bytes(self, n_ubatch: int = 512, n_ctx: int = 4096, flash_attn: bool = True) -> int:
        """Rough size of llama.cpp's compute buffer for one batch of n_ubatch tokens."""
        logits = n_ubatch * self.n_vocab * 4
        ff = self.n_ff * (self.n_experts_used or 1)
        ffn = n_ubatch * ff * 4 * 3
        attn = 0 if flash_attn else n_ubatch * _pad(n_ctx) * 4 * max(1, self.n_embd // 128)
        return max(logits, ffn, attn) + n_ubatch * self.n_embd * 4 * 4


def layout(header: GGUFHeader, tensors: list[Tensor], cache_type: str = "f16") -> Layout:
    h = header
    arch = h.arch
    n_layer_all = int(h.get("block_count", 0))
    nextn = int(h.get("nextn_predict_layers", 0) or 0)
    n_layer = n_layer_all - nextn
    n_embd = int(h.get("embedding_length", 0))
    notes = []
    if nextn:
        notes.append(f"{nextn} multi-token-prediction layer(s) are not loaded by default")

    layer_bytes = [0] * n_layer
    expert_bytes = [0] * n_layer
    input_bytes = output_bytes = 0
    has_output = False
    embd_bytes = 0
    for t in tensors:
        m = _BLOCK.match(t.name)
        if m:
            il = int(m.group(1))
            if il < n_layer:
                layer_bytes[il] += t.nbytes
                if "_exps" in t.name:
                    expert_bytes[il] += t.nbytes
        elif t.name.startswith(("token_embd", "per_layer_token_embd")):
            input_bytes += t.nbytes
            if t.name == "token_embd.weight":
                embd_bytes = t.nbytes
        else:
            output_bytes += t.nbytes
            has_output |= t.name == "output.weight"
    if not has_output:
        output_bytes += embd_bytes  # tied embeddings: llama.cpp loads a second copy as the output head

    n_vocab = int(h.get("vocab_size", 0) or 0) or len(h.metadata.get("tokenizer.ggml.tokens") or [])
    n_ff = _per_layer(h.get("feed_forward_length", 0) or 0, max(n_layer, 1))
    n_experts = int(h.get("expert_count", 0) or 0)
    n_used = int(h.get("expert_used_count", 0) or 0)
    ff = int(h.get("expert_feed_forward_length", 0) or 0) if n_experts else max(n_ff)

    # which layers are recurrent (no KV cache), which use a sliding window
    head_kv = _per_layer(h.get("attention.head_count_kv", h.get("attention.head_count", 0)) or 0, n_layer)
    head_q = _per_layer(h.get("attention.head_count", 0) or 0, n_layer)
    recurrent = [False] * n_layer
    if isinstance(h.get("attention.recurrent_layers"), list):
        recurrent = [bool(x) for x in h.get("attention.recurrent_layers")[:n_layer]]
    elif arch in RECURRENT_INTERVAL:
        every = int(h.get("full_attention_interval", RECURRENT_INTERVAL[arch]))
        recurrent = [(i + 1) % every != 0 for i in range(n_layer)]
    elif arch in PURE_RECURRENT or not any(head_q):
        recurrent = [True] * n_layer
    recurrent = [r or kv == 0 for r, kv in zip(recurrent, head_kv)]

    n_swa = int(h.get("attention.sliding_window", 0) or 0)
    pattern = h.get("attention.sliding_window_pattern")
    if isinstance(pattern, list):
        swa = [bool(x) for x in pattern[:n_layer]] + [False] * max(0, n_layer - len(pattern))
    else:
        default_pattern, default_window = SWA_DEFAULTS.get(arch, (0, 0))
        n_swa = n_swa or default_window
        if arch == "llama4" and h.get("attention.sliding_window") == 0:
            n_swa = 0
        pattern = int(pattern) if pattern is not None else default_pattern
        swa = [bool(n_swa) and pattern > 0 and i % pattern < pattern - 1 for i in range(n_layer)]
    if not n_swa:
        swa = [False] * n_layer

    is_mla = h.get("attention.key_length_mla") is not None
    elt = CACHE_TYPES[cache_type]
    kv_full, kv_swa = [0.0] * n_layer, [0.0] * n_layer
    for i in range(n_layer):
        if recurrent[i]:
            continue
        default = n_embd // head_q[i] if head_q[i] else 0
        k = int(h.get("attention.key_length_swa" if swa[i] else "attention.key_length", 0) or h.get("attention.key_length", 0) or default)
        v = int(
            h.get("attention.value_length_swa" if swa[i] else "attention.value_length", 0) or h.get("attention.value_length", 0) or default
        )
        per_token = head_kv[i] * (k + (0 if is_mla else v)) * elt
        (kv_swa if swa[i] else kv_full)[i] = per_token
    if any(swa):
        notes.append(f"{sum(swa)} of {n_layer} layers use a {n_swa}-token sliding window (smaller cache)")
    if is_mla:
        notes.append("MLA attention: compressed KV cache")

    # recurrent state, float32, one sequence (llama_hparams::n_embd_r / n_embd_s)
    rec = [0] * n_layer
    if any(recurrent):
        d_conv = int(h.get("ssm.conv_kernel", 0) or 0)
        d_inner = int(h.get("ssm.inner_size", 0) or 0)
        d_state = int(h.get("ssm.state_size", 0) or 0)
        n_group = int(h.get("ssm.group_count", 0) or 0)
        wkv = int(h.get("wkv.head_size", 0) or 0)
        l_cache = int(h.get("shortconv.l_cache", 0) or 0)
        if wkv:
            r, s = int(h.get("token_shift_count", 2) or 2) * n_embd, n_embd * wkv
        elif l_cache:
            r, s = n_embd * (l_cache - 1), d_state * d_inner
        else:
            r, s = max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state), d_state * d_inner
        rec = [(r + s) * 4 if recurrent[i] else 0 for i in range(n_layer)]
        notes.append(f"{sum(recurrent)} of {n_layer} layers are recurrent (fixed-size state, no KV cache)")

    return Layout(
        arch,
        n_layer,
        int(h.get("context_length", 0) or 0),
        n_embd,
        n_vocab,
        ff,
        layer_bytes,
        expert_bytes,
        input_bytes,
        output_bytes,
        kv_full,
        kv_swa,
        n_swa,
        rec,
        n_experts,
        n_used,
        notes,
    )


@dataclass
class Plan:
    verdict: str  # "gpu", "moe", "partial", "cpu", "no"
    gpu_layers: int  # llama.cpp -ngl value (n_layer + 1 means everything)
    cpu_moe: int  # llama.cpp --n-cpu-moe value
    vram: list[int]  # bytes used per GPU
    ram: int  # bytes of weights and cache in system RAM
    total: int
    kv: int
    compute: int
    max_ctx: int  # largest context with this placement that still fits, 0 if unknown


def plan(lay: Layout, n_ctx: int, gpus: list[int], ram: int, n_ubatch: int = 512, flash_attn: bool = True) -> Plan:
    """Place the model like llama.cpp: layers from the last one down on the GPUs, the rest in RAM.

    `gpus` are the usable bytes per GPU (free memory minus a safety margin).
    """
    n = lay.n_layer
    compute = lay.compute_bytes(n_ubatch, n_ctx, flash_attn)
    kv_layer = [lay.kv_bytes(n_ctx, n_ubatch, range(i, i + 1)) for i in range(n)]
    kv_total = int(sum(kv_layer))
    total = sum(lay.layer_bytes) + lay.input_bytes + lay.output_bytes + kv_total + compute

    def place(n_gpu: int, cpu_moe: int) -> list[int] | None:
        """VRAM per GPU when the last n_gpu layers are offloaded, or None if it does not fit."""
        if not gpus:
            return None
        caps = [g - compute for g in gpus]
        if min(caps) < 0:
            return None
        first = n - min(n_gpu, n)
        layers = [lay.layer_bytes[i] + kv_layer[i] - (lay.expert_bytes[i] if i < cpu_moe else 0) for i in range(first, n)]
        need_out = lay.output_bytes if n_gpu > n else 0
        # llama.cpp's default split: layers in order, in proportion to each GPU's free memory
        used = [0] * len(caps)
        bounds, acc = [], 0.0
        for g in gpus:
            acc += g / sum(gpus)
            bounds.append(acc)
        for j, b in enumerate(layers):
            frac = j / len(layers)
            used[next(k for k, edge in enumerate(bounds) if frac < edge or k == len(bounds) - 1)] += b
        used[-1] += need_out
        if any(u > c for u, c in zip(used, caps)):
            return None
        return [u + compute for u in used]

    def ram_for(n_gpu: int, cpu_moe: int) -> int:
        first = n - min(n_gpu, n)
        cpu_layers = sum(lay.layer_bytes[:first]) + sum(kv_layer[:first])
        experts = sum(lay.expert_bytes[i] for i in range(first, min(cpu_moe, n)))
        out = 0 if n_gpu > n else lay.output_bytes
        host_compute = compute if n_gpu <= n else 0
        return int(lay.input_bytes + cpu_layers + experts + out + host_compute)

    def result(verdict, n_gpu, cpu_moe, vram):
        r = ram_for(n_gpu, cpu_moe)
        if r > ram:
            return None
        return Plan(verdict, n_gpu, cpu_moe, vram, r, total, kv_total, compute, 0)

    full = place(n + 1, 0)
    if full is not None and (r := result("gpu", n + 1, 0, full)):
        return r
    if any(lay.expert_bytes) and gpus:
        for k in range(1, n + 1):  # keep the experts of the first k layers in RAM
            vram = place(n + 1, k)
            if vram is not None:
                r = result("moe", n + 1, k, vram)
                if r:
                    return r
                break
    for n_gpu in range(n, 0, -1):
        vram = place(n_gpu, 0)
        if vram is not None:
            r = result("partial", n_gpu, 0, vram)
            if r:
                return r
            break
    cpu_ram = ram_for(0, 0)
    if cpu_ram <= ram:
        return Plan("cpu", 0, 0, [0] * len(gpus), cpu_ram, total, kv_total, compute, 0)
    return Plan("no", 0, 0, [0] * len(gpus), cpu_ram, total, kv_total, compute, 0)


def max_context(lay: Layout, gpus: list[int], ram: int, n_ubatch: int = 512, step: int = 1024) -> int:
    """Largest context (multiple of `step`, up to the training length) that fits fully on the GPUs."""
    limit = lay.n_ctx_train or 131072
    best, lo, hi = 0, 1, max(1, limit // step)
    while lo <= hi:
        mid = (lo + hi) // 2
        if plan(lay, mid * step, gpus, ram, n_ubatch).verdict == "gpu":
            best, lo = mid * step, mid + 1
        else:
            hi = mid - 1
    return best
