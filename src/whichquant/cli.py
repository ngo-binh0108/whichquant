"""whichquant: which GGUF quant of a model fits your GPU and RAM?"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from whichquant import __version__, hardware, hub
from whichquant.estimate import CACHE_TYPES, GiB, layout, max_context, plan

VERDICTS = {
    "gpu": "all on GPU",
    "moe": "GPU + experts in RAM",
    "partial": "split GPU/RAM",
    "cpu": "RAM only",
    "no": "does not fit",
}


def parse_sizes(text: str) -> list[float]:
    return [float(x) for x in re.split(r"[,+ ]+", text.strip()) if x]


def gib(n: float) -> str:
    return f"{n / GiB:.1f}"


def ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def flags(p, n_ctx: int, n_layer: int) -> str:
    if p.verdict == "no":
        return ""
    ngl = "99" if p.gpu_layers > n_layer else str(p.gpu_layers)
    parts = [f"-ngl {ngl}", f"-c {n_ctx}"]
    if p.cpu_moe:
        parts.append(f"--n-cpu-moe {p.cpu_moe}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="whichquant",
        description="Which GGUF quant fits your GPU? Reads only the file headers from Hugging Face, so nothing big is downloaded.",
    )
    ap.add_argument("model", help="Hugging Face repo (unsloth/Qwen3-8B-GGUF), its URL, or a local .gguf file")
    ap.add_argument("-c", "--ctx", type=int, default=8192, help="context length in tokens (default 8192)")
    ap.add_argument("--gpu", help="GPU memory in GB, e.g. 24 or 16,16 for two cards (default: detect NVIDIA / Apple)")
    ap.add_argument("--ram", type=float, help="system RAM in GB (default: detect)")
    ap.add_argument("--kv", default="f16", choices=sorted(CACHE_TYPES), help="KV cache type, llama.cpp -ctk/-ctv (default f16)")
    ap.add_argument("--reserve", type=float, default=1.0, help="GB to keep free on each GPU (default 1, like llama.cpp --fit)")
    ap.add_argument("--ubatch", type=int, default=512, help="llama.cpp -ub, sets the compute buffer size (default 512)")
    ap.add_argument("-q", "--quant", help="only show quants whose name contains this, with a full breakdown")
    ap.add_argument("--vision", action="store_true", help="also count the repo's smallest vision projector (mmproj file)")
    ap.add_argument("--revision", default="main", help="repo branch or commit")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    ap.add_argument("-V", "--version", action="version", version=f"whichquant {__version__}")
    args = ap.parse_args(argv)

    if args.ctx < 1:
        ap.error("--ctx must be positive")

    # hardware
    devices, ram_total, _ = hardware.detect()
    unified = bool(devices) and devices[0].name.startswith("Apple")
    if args.gpu is not None:
        devices = [hardware.Device(f"GPU {i}", int(g * GiB), int(g * GiB)) for i, g in enumerate(parse_sizes(args.gpu)) if g > 0]
        unified = False
    if args.ram is not None:
        ram_total = int(args.ram * GiB)
    reserve = int(args.reserve * GiB)
    gpus = [max(0, d.free - reserve) for d in devices]
    ram = int(ram_total * 0.9) - (sum(gpus) if unified else 0)

    # model files
    source = args.model.strip()
    url = re.match(r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([^/]+/[^/?#]+)", source)
    repo = url.group(1) if url else source
    projector = None
    if Path(source).is_file():
        if args.vision:
            ap.error("--vision needs a Hugging Face repo, to find the mmproj file")
        models, repo = [hub.local_model(source)], None
    else:
        if repo.count("/") != 1:
            ap.error(f"{source!r} is neither a file nor a Hugging Face repo like org/name")
        files = hub.list_repo(repo, args.revision)
        models = hub.group_models(files)
        if args.vision:
            projector = hub.smallest_projector(files)
            if projector is None:
                print(f"{repo} has no mmproj file, so no vision projector is counted", file=sys.stderr)
        if args.quant:
            models = [m for m in models if args.quant.lower() in m.name.lower()]
        if not models:
            print(f"no GGUF files{' matching ' + args.quant if args.quant else ''} in {repo}", file=sys.stderr)
            return 1
        print(f"reading {len(models)} GGUF header(s) from {repo} ...", file=sys.stderr)
        hub.load_repo_headers(models, repo, args.revision)

    # llama.cpp loads the projector on the first GPU, next to the model
    projector_gpu = bool(projector) and bool(gpus) and gpus[0] >= projector[1]
    if projector_gpu:
        gpus[0] -= projector[1]
    elif projector:
        ram -= projector[1]

    rows = []
    for m in models:
        lay = layout(m.header, m.tensors, args.kv)
        p = plan(lay, args.ctx, gpus, ram, args.ubatch)
        if projector:
            p.total += projector[1]
            if projector_gpu:
                p.vram[0] += projector[1]
            else:
                p.ram += projector[1]
        best_ctx = max_context(lay, gpus, ram, args.ubatch) if p.verdict == "gpu" else 0
        rows.append((m, lay, p, best_ctx))

    lay0 = rows[0][1]
    if args.json:
        print(
            json.dumps(
                {
                    "model": repo or source,
                    "arch": lay0.arch,
                    "layers": lay0.n_layer,
                    "context": args.ctx,
                    "kv_cache": args.kv,
                    "gpus": [{"name": d.name, "usable_bytes": g} for d, g in zip(devices, gpus)],
                    "ram_bytes": ram,
                    "vision_projector": {"file": projector[0], "bytes": projector[1], "on_gpu": projector_gpu} if projector else None,
                    "quants": [
                        {
                            "quant": m.quant,
                            "file": m.name,
                            "file_bytes": m.size,
                            "needed_bytes": p.total,
                            "kv_bytes": p.kv,
                            "fit": p.verdict,
                            "gpu_layers": p.gpu_layers,
                            "n_cpu_moe": p.cpu_moe,
                            "vram_bytes": p.vram,
                            "ram_bytes": p.ram,
                            "max_gpu_context": best_ctx or None,
                        }
                        for m, _, p, best_ctx in rows
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"\n{repo or source}")
    moe = f" | MoE {lay0.n_experts} experts, {lay0.n_experts_used} active" if lay0.n_experts else ""
    print(f"{lay0.arch} | {lay0.n_layer} layers{moe} | trained context {ctx_label(lay0.n_ctx_train)} | context {args.ctx} | KV {args.kv}")
    for note in lay0.notes:
        print(f"  {note}")
    if projector:
        print(f"vision projector: {projector[0]}, {gib(projector[1])} GiB on {devices[0].name if projector_gpu else 'RAM'}, counted below")
    if devices:
        cards = " + ".join(f"{d.name} {gib(g)}" for d, g in zip(devices, gpus))
        print(f"GPU usable: {cards} GiB | RAM usable: {gib(ram)} GiB")
    else:
        print(f"no GPU found (use --gpu 24, or --gpu 16,16) | RAM usable: {gib(ram)} GiB")
    print()

    header = f"{'quant':<14}{'file GiB':>9}{'needs GiB':>11}  {'fit':<22}{'max ctx':>8}  llama.cpp"
    print(header)
    print("-" * len(header))
    for m, lay, p, best_ctx in rows:
        fit = VERDICTS[p.verdict]
        if p.verdict == "partial":
            fit = f"{p.gpu_layers}/{lay.n_layer + 1} layers on GPU"
        best = ctx_label(best_ctx) if best_ctx else ""
        print(f"{m.quant:<14}{gib(m.size):>9}{gib(p.total):>11}  {fit:<22}{best:>8}  {flags(p, args.ctx, lay.n_layer)}")

    on_gpu = [r for r in rows if r[2].verdict == "gpu"]
    print()
    moe = [r for r in rows if r[2].verdict == "moe"]
    partial = [r for r in rows if r[2].verdict == "partial"]
    if on_gpu:
        best = max(on_gpu, key=lambda r: r[0].size)
        print(f"Largest quant that runs fully on your GPU: {best[0].quant} ({best[0].name})")
    if moe:
        best = max(moe, key=lambda r: r[0].size)
        print(f"With some MoE experts in RAM (usually still fast): up to {best[0].quant}")
    elif partial and not on_gpu:
        best = max(partial, key=lambda r: r[0].size)
        print(f"Nothing fits fully on the GPU. Largest quant split between GPU and RAM: {best[0].quant} (slower)")
    elif not on_gpu and any(r[2].verdict == "cpu" for r in rows):
        print("Without a GPU, the model runs from RAM on the CPU (slow for large models).")
    if args.quant:
        for m, lay, p, _ in rows:
            print(
                f"\n{m.name}: weights {gib(sum(lay.layer_bytes) + lay.input_bytes + lay.output_bytes)} GiB "
                f"(embeddings {gib(lay.input_bytes)} in RAM), KV cache {gib(p.kv)}, compute ~{gib(p.compute)}"
            )
            if p.verdict == "no":
                print(f"  needs {gib(p.total)} GiB, more than your GPU and RAM together")
                continue
            for d, used in zip(devices, p.vram):
                print(f"  {d.name}: {gib(used)} GiB")
            print(f"  RAM: {gib(p.ram)} GiB")
            file = m.paths[0].split("/")[-1]
            ngl = 99 if p.gpu_layers > lay.n_layer else p.gpu_layers
            kv = f" -ctk {args.kv} -ctv {args.kv}" if args.kv != "f16" else ""
            print(f"  llama-server -m {file} {flags(p, args.ctx, lay.n_layer)}{kv}")
            print(
                f"  koboldcpp --model {file} --gpulayers {ngl} --contextsize {args.ctx}"
                f"{f' --moecpu {p.cpu_moe}' if p.cpu_moe else ''}{f' --quantkv {args.kv}' if kv else ''}"
            )
    print("\nWeights are exact; the KV cache follows llama.cpp's sizing; the compute buffer is an estimate.")
    if any(p.verdict in ("moe", "partial") for _, _, p, _ in rows):
        print("Recent llama.cpp and KoboldCpp builds tune -ngl and --n-cpu-moe themselves (--fit); the flags above are a starting point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
