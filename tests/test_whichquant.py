import io
import json
import struct

import pytest

from whichquant import cli, hub
from whichquant.estimate import GiB, layout, max_context, plan
from whichquant.gguf import GGML_TYPES, GGUFError, SkippedStrings, read_header, read_local

# --- a minimal GGUF writer -------------------------------------------------


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> tuple[int, bytes]:
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        return 8, _str(v)
    if isinstance(v, list):
        itype = _value(v[0])[0] if v else 4
        return 9, struct.pack("<IQ", itype, len(v)) + b"".join(_value(x)[1] for x in v)
    raise TypeError(v)


def gguf_bytes(metadata: dict, tensors: list[tuple[str, tuple, int]]) -> bytes:
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        vtype, payload = _value(value)
        out += _str(key) + struct.pack("<I", vtype) + payload
    for name, shape, type_id in tensors:
        out += _str(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", type_id, 0)
    return out


def model(arch="llama", n_layer=4, n_embd=256, n_head=8, n_kv=2, vocab=1000, ff=512, extra=None, tied=False, experts=0, qtype=12):
    md = {
        "general.architecture": arch,
        f"{arch}.block_count": n_layer,
        f"{arch}.context_length": 4096,
        f"{arch}.embedding_length": n_embd,
        f"{arch}.feed_forward_length": ff,
        f"{arch}.attention.head_count": n_head,
        f"{arch}.attention.head_count_kv": n_kv,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab)],
    }
    if experts:
        md[f"{arch}.expert_count"] = experts
        md[f"{arch}.expert_used_count"] = 2
        md[f"{arch}.expert_feed_forward_length"] = ff
    md.update(extra or {})
    tensors = [("token_embd.weight", (n_embd, vocab), qtype), ("output_norm.weight", (n_embd,), 0)]
    if not tied:
        tensors.append(("output.weight", (n_embd, vocab), qtype))
    for i in range(n_layer):
        tensors += [
            (f"blk.{i}.attn_q.weight", (n_embd, n_embd), qtype),
            (f"blk.{i}.attn_norm.weight", (n_embd,), 0),
        ]
        if experts:
            tensors.append((f"blk.{i}.ffn_up_exps.weight", (n_embd, ff, experts), qtype))
        else:
            tensors.append((f"blk.{i}.ffn_up.weight", (n_embd, ff), qtype))
    return md, tensors


def header_of(md, tensors):
    data = gguf_bytes(md, tensors)
    pos = 0

    def read(n):
        nonlocal pos
        chunk = data[pos : pos + n]
        pos += n
        return chunk

    return read_header(read)


# --- parsing ---------------------------------------------------------------


def test_header_roundtrip_and_tensor_sizes(tmp_path):
    md, tensors = model(extra={"llama.rope.freq_base": 10000.0, "llama.flag": True, "general.name": "tiny"})
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes(md, tensors))
    h = read_local(path)
    assert h.arch == "llama" and h.get("block_count") == 4 and h.metadata["general.name"] == "tiny"
    assert h.metadata["llama.flag"] is True and h.get("rope.freq_base") == pytest.approx(10000.0)
    assert len(h.tensors) == len(tensors)
    q = next(t for t in h.tensors if t.name == "blk.0.attn_q.weight")
    assert q.type_name == "Q4_K" and q.nbytes == 256 * 256 // 256 * 144
    assert next(t for t in h.tensors if t.name == "output_norm.weight").nbytes == 256 * 4
    assert h.tensor_info_offset < h.header_size


def test_long_string_arrays_are_skipped_but_counted():
    md, tensors = model(vocab=5000)
    h = header_of(md, tensors)
    vocab = h.metadata["tokenizer.ggml.tokens"]
    assert isinstance(vocab, SkippedStrings) and len(vocab) == 5000
    assert len(h.tensors) == len(tensors)


def test_rejects_non_gguf():
    with pytest.raises(GGUFError):
        read_header(io.BytesIO(b"NOPE" + bytes(20)).read)
    with pytest.raises(GGUFError):
        read_header(io.BytesIO(b"GGUF").read)  # truncated


def test_type_table_matches_gguf_package():
    gguf = pytest.importorskip("gguf")
    for qtype, (block, size) in gguf.GGML_QUANT_SIZES.items():
        assert GGML_TYPES[int(qtype)][1:] == (block, size), qtype.name


# --- reading only the tensor table of the other quants ------------------------


def test_tensor_table_found_next_to_reference(monkeypatch):
    md, tensors = model()
    ref = header_of(md, tensors)
    # another quant: an extra metadata key shifts its table, and its tensors are Q8_0
    md2 = {**md, "quantize.imatrix.file": "imatrix.dat"}
    other = gguf_bytes(md2, [(n, s, 8 if t == 12 else t) for n, s, t in tensors])
    monkeypatch.setattr(hub, "fetch_range", lambda url, start, end: other[start : end + 1])
    h = hub._tensor_table_near("x", ref)
    assert h is not None and [t.name for t in h.tensors] == [t.name for t in ref.tensors]
    assert {t.type_name for t in h.tensors} == {"Q8_0", "F32"}


# --- repo listing --------------------------------------------------------------


def test_quant_labels():
    assert hub.quant_label("Llama-3.3-70B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert hub.quant_label("Qwen3-8B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert hub.quant_label("gpt-oss-20b-F16.gguf") == "F16"
    assert hub.quant_label("model-IQ2_XXS-00001-of-00002.gguf") == "IQ2_XXS"
    assert hub.quant_label("weird.gguf") == "weird"


def test_group_models_joins_splits_and_skips_projectors():
    files = [
        ("Q8_0/m-Q8_0-00001-of-00002.gguf", 10),
        ("Q8_0/m-Q8_0-00002-of-00002.gguf", 5),
        ("m-Q4_K_M.gguf", 8),
        ("mmproj-F16.gguf", 1),
        ("README.md", 1),
        ("dup/m-Q4_K_M.gguf", 8),
    ]
    models = hub.group_models(files)
    assert [(m.quant, m.size, len(m.paths)) for m in models] == [("Q4_K_M", 8, 1), ("Q8_0", 15, 2)]


# --- memory model ----------------------------------------------------------------


def test_kv_cache_matches_hand_calculation():
    md, tensors = model(n_layer=4, n_embd=256, n_head=8, n_kv=2)
    lay = layout(header_of(md, tensors), header_of(md, tensors).tensors)
    # 4 layers * 2 kv heads * (32 + 32) dims * 2 bytes, per token; 1000 tokens pad to 1024 cells
    assert lay.kv_bytes(1000) == 4 * 2 * 64 * 2 * 1024
    assert layout(header_of(md, tensors), header_of(md, tensors).tensors, "q8_0").kv_bytes(1024) == pytest.approx(
        4 * 2 * 64 * 1024 * 34 / 32
    )


def test_gemma3_sliding_window_default_pattern():
    md, tensors = model(arch="gemma3", n_layer=12, extra={"gemma3.attention.sliding_window": 1024})
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    swa = [i for i in range(12) if lay.kv_swa[i]]
    assert swa == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # every 6th layer is full attention
    per_token = 2 * 64 * 2
    assert lay.kv_bytes(32768) == 2 * per_token * 32768 + 10 * per_token * 1536  # window + ubatch, padded


def test_explicit_swa_pattern_array():
    pattern = [True, False, True, False]
    md, tensors = model(
        arch="newarch", extra={"newarch.attention.sliding_window": 512, "newarch.attention.sliding_window_pattern": pattern}
    )
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    assert [bool(x) for x in lay.kv_swa] == pattern


def test_hybrid_recurrent_layers_have_no_kv():
    extra = {
        "qwen3next.full_attention_interval": 4,
        "qwen3next.ssm.conv_kernel": 4,
        "qwen3next.ssm.inner_size": 64,
        "qwen3next.ssm.state_size": 16,
        "qwen3next.ssm.group_count": 2,
    }
    md, tensors = model(arch="qwen3next", n_layer=8, extra=extra)
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    assert [i for i in range(8) if lay.kv_full[i]] == [3, 7]
    state = (3 * (64 + 2 * 2 * 16) + 16 * 64) * 4
    assert lay.recurrent_bytes == [0 if i in (3, 7) else state for i in range(8)]


def test_head_count_kv_zero_marks_recurrent_layer():
    md, tensors = model(arch="lfm2", n_layer=4, extra={"lfm2.attention.head_count_kv": [0, 2, 0, 2], "lfm2.shortconv.l_cache": 3})
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    assert [i for i in range(4) if lay.kv_full[i]] == [1, 3]
    assert lay.recurrent_bytes[0] == 256 * 2 * 4


def test_mla_has_no_v_cache():
    extra = {
        "deepseek2.attention.key_length": 576,
        "deepseek2.attention.value_length": 512,
        "deepseek2.attention.key_length_mla": 192,
        "deepseek2.attention.value_length_mla": 128,
    }
    md, tensors = model(arch="deepseek2", n_kv=1, extra=extra)
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    assert lay.kv_full[0] == 576 * 2


def test_tied_embeddings_and_mtp_layers():
    md, tensors = model(n_layer=5, tied=True, extra={"llama.nextn_predict_layers": 1})
    h = header_of(md, tensors)
    lay = layout(h, h.tensors)
    embd = 256 * 1000 // 256 * 144
    assert lay.n_layer == 4 and len(lay.layer_bytes) == 4
    assert lay.input_bytes == embd and lay.output_bytes == embd + 256 * 4  # second copy as output head


def _lay(**kw):
    md, tensors = model(**kw)
    h = header_of(md, tensors)
    return layout(h, h.tensors)


def test_plan_verdicts():
    lay = _lay(n_layer=8, n_embd=1024, ff=4096, vocab=2000)
    total = plan(lay, 2048, [100 * GiB], 100 * GiB).total
    assert plan(lay, 2048, [100 * GiB], 100 * GiB).verdict == "gpu"
    half = plan(lay, 2048, [int(total * 0.6)], 100 * GiB)
    assert half.verdict == "partial" and 0 < half.gpu_layers <= 8
    assert plan(lay, 2048, [], 100 * GiB).verdict == "cpu"
    assert plan(lay, 2048, [], 1).verdict == "no"


def test_plan_keeps_experts_in_ram_before_splitting_layers():
    lay = _lay(n_layer=8, n_embd=1024, ff=1024, experts=16, vocab=2000)
    full = plan(lay, 2048, [100 * GiB], 100 * GiB)
    p = plan(lay, 2048, [int(full.total * 0.5)], 100 * GiB)
    assert p.verdict == "moe" and p.gpu_layers == 9 and 0 < p.cpu_moe <= 8
    assert p.vram[0] <= int(full.total * 0.5)


def test_plan_splits_layers_across_gpus():
    lay = _lay(n_layer=8, n_embd=1024, ff=4096, vocab=2000)
    p = plan(lay, 2048, [10 * GiB, 10 * GiB], 100 * GiB)
    assert p.verdict == "gpu" and abs(p.vram[0] - p.vram[1]) < p.vram[1] * 0.5


def test_max_context_grows_with_memory():
    lay = _lay(n_layer=8, n_embd=1024, ff=4096, vocab=2000, extra={"llama.context_length": 65536})
    small = max_context(lay, [int(0.1 * GiB)], 100 * GiB)
    big = max_context(lay, [int(0.5 * GiB)], 100 * GiB)
    assert 0 < small < big <= lay.n_ctx_train


# --- command line ------------------------------------------------------------------


def test_cli_on_local_file(tmp_path, capsys):
    md, tensors = model(n_layer=8, n_embd=1024, ff=4096, vocab=2000)
    path = tmp_path / "tiny-Q4_K_M.gguf"
    path.write_bytes(gguf_bytes(md, tensors))
    assert cli.main([str(path), "--gpu", "8", "--ram", "16", "-c", "2048"]) == 0
    out = capsys.readouterr().out
    assert "Q4_K_M" in out and "all on GPU" in out and "-ngl 99 -c 2048" in out


def test_cli_json(tmp_path, capsys):
    md, tensors = model()
    path = tmp_path / "tiny-Q8_0.gguf"
    path.write_bytes(gguf_bytes(md, tensors))
    assert cli.main([str(path), "--gpu", "0.001", "--ram", "16", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["quants"][0]["fit"] in ("partial", "cpu") and data["layers"] == 4
