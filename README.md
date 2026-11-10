# whichquant

[![tests](https://github.com/ngo-binh0108/whichquant/actions/workflows/tests.yml/badge.svg)](https://github.com/ngo-binh0108/whichquant/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Which GGUF quant should you download? whichquant checks every quant in a Hugging Face repo against your GPU and RAM, before you download 40 GB.**

```bash
pip install git+https://github.com/ngo-binh0108/whichquant
whichquant unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF -c 32768
```

```
unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
qwen3moe | 48 layers | MoE 128 experts, 8 active | trained context 256k | context 32768 | KV f16
GPU usable: GPU 0 15.0 GiB | RAM usable: 28.8 GiB

quant          file GiB  needs GiB  fit                    max ctx  llama.cpp
-----------------------------------------------------------------------------
UD-IQ2_M           10.1       13.4  all on GPU                 50k  -ngl 99 -c 32768
Q2_K               10.5       13.8  all on GPU                 45k  -ngl 99 -c 32768
UD-Q2_K_XL         11.0       14.3  all on GPU                 41k  -ngl 99 -c 32768
UD-IQ3_XXS         12.0       15.3  GPU + experts in RAM            -ngl 99 -c 32768 --n-cpu-moe 1
Q3_K_M             13.7       17.0  GPU + experts in RAM            -ngl 99 -c 32768 --n-cpu-moe 7
Q4_K_M             17.3       20.6  GPU + experts in RAM            -ngl 99 -c 32768 --n-cpu-moe 16
Q5_K_M             20.2       23.5  GPU + experts in RAM            -ngl 99 -c 32768 --n-cpu-moe 21
Q8_0               30.3       33.6  GPU + experts in RAM            -ngl 99 -c 32768 --n-cpu-moe 31
BF16               56.9       60.2  does not fit
...

Largest quant that runs fully on your GPU: UD-Q2_K_XL (Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf)
With some MoE experts in RAM (usually still fast): up to UD-Q8_K_XL
```

(A 16 GB GPU with 32 GB of RAM; some rows are left out.)

## Why

The file size is only part of what a model needs. The KV cache for your context length can add
several GB, and how big it gets depends on the architecture. Gemma 3 keeps most layers to a
1024-token window, Qwen3-Next only caches 12 of its 48 layers, and DeepSeek compresses its cache
with MLA. A quant that "should fit" by file size often doesn't once you set a long context.

whichquant reads the **header** of each GGUF file over HTTP range requests: the metadata and the
tensor table, a few MB out of tens of GB. It reads the full metadata once per repo, and for every
other quant only fetches its tensor table. That's enough to know:

- the **exact size of every layer**, including which parts are MoE experts;
- the **KV cache** for your context, following llama.cpp's rules for grouped-query attention,
  sliding windows, MLA, hybrid recurrent layers and multi-token-prediction layers (which llama.cpp skips);
- **how to place it**: all on the GPU, MoE experts in RAM (`--n-cpu-moe`), or some layers on the CPU (`-ngl`);
- the **largest context** that still fits fully on your GPU.

## Usage

```bash
whichquant unsloth/gemma-3-27b-it-GGUF                # detect NVIDIA GPUs and RAM
whichquant unsloth/Llama-3.3-70B-Instruct-GGUF --gpu 16,16 --ram 64   # two 16 GB cards
whichquant bartowski/Qwen_Qwen3-32B-GGUF -c 65536 --kv q8_0           # long context, quantised cache
whichquant unsloth/GLM-4.5-Air-GGUF -q Q4_K_M         # one quant, with a full breakdown and commands
whichquant ./models/my-model.gguf                     # a local file
whichquant unsloth/Qwen3-8B-GGUF --json               # machine-readable
whichquant ggml-org/gemma-3-4b-it-GGUF --vision       # also count the vision projector
```

With `-q` you also get the memory per GPU and ready-to-run commands:

```
Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf: weights 16.5 GiB (embeddings 0.2 in RAM), KV cache 3.0, compute ~0.3
  GPU 0: 14.8 GiB
  RAM: 5.0 GiB
  llama-server -m Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf -ngl 99 -c 32768 --n-cpu-moe 15
  koboldcpp --model Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf --gpulayers 99 --contextsize 32768 --moecpu 15
```

| Option | Default | Meaning |
|---|---|---|
| `-c, --ctx` | `8192` | context length in tokens |
| `--gpu` | detect | GPU memory in GB per card, e.g. `24` or `16,16` |
| `--ram` | detect | system RAM in GB |
| `--kv` | `f16` | KV cache type (`q8_0`, `q4_0`, ... as llama.cpp `-ctk`/`-ctv`) |
| `--reserve` | `1` | GB kept free on each GPU, like llama.cpp `--fit` |
| `--ubatch` | `512` | llama.cpp `-ub`, which sets the compute buffer size |
| `-q, --quant` | all | only quants whose name contains this, with a breakdown |
| `--vision` | off | also count the repo's smallest vision projector (`mmproj` file) |
| `--revision` | `main` | repo branch or commit |
| `--json` | off | JSON output |

GPUs are detected with `nvidia-smi`, using the **free** memory, so close other GPU programs first.
On Apple Silicon the GPU limit is set to macOS's default share of unified memory. For AMD or
Intel GPUs, pass `--gpu`. For gated repos (Llama, Gemma), accept the licence on Hugging Face and
set `HF_TOKEN`.

It has no dependencies beyond the Python standard library.

## How accurate is it?

- **Weights: exact.** They are summed from the tensor table, and match the file sizes.
- **KV cache and recurrent state:** computed with llama.cpp's sizing rules: cells padded to 256,
  the sliding-window cache of `window + ubatch` cells, no V cache with MLA, and float32 state for
  Mamba / gated-delta-net layers. These are checked against hand calculations in the tests.
- **Compute buffer: an estimate** (usually 0.3 to 0.6 GB with flash attention), plus the 1 GB
  margin per GPU.
- **Placement** follows llama.cpp's defaults: token embeddings in RAM, layers split across GPUs in
  proportion to their free memory, and the output layer on the last GPU.

Not counted: vision projectors (`mmproj` files) unless you pass `--vision`, speculative-decoding draft models, and several
parallel sequences (`-np`) with a non-unified cache. Recent llama.cpp and KoboldCpp builds pick
`-ngl` and `--n-cpu-moe` themselves at load time (`--fit`). whichquant answers the earlier
question: which file to download.

If an estimate is off for your setup, please open an issue with the model, your command and
llama.cpp's log lines (`model buffer size`, `KV buffer size`, `compute buffer size`).

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
