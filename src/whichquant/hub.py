"""Find the GGUF files in a Hugging Face repo and read their headers over HTTP."""

from __future__ import annotations

import json
import os
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from whichquant import __version__
from whichquant.gguf import GGUFError, GGUFHeader, read_counts, read_header, read_local, read_tensor_table

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

_SPLIT = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")
_PART = re.compile(r"^(?P<stem>.+\.gguf)\.part(?P<no>\d+)of(?P<count>\d+)$")
_QUANT = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|TQ\d_\d|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32|FP16|FP8))(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _request(url: str, headers: dict[str, str] | None = None):
    headers = {"User-Agent": f"whichquant/{__version__}", **(headers or {})}
    token = _token()
    if token and url.startswith(HF_ENDPOINT):
        headers["Authorization"] = f"Bearer {token}"
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise SystemExit(
                f"{url}: access denied (HTTP {err.code}). For a gated or private repo, accept its licence "
                "on huggingface.co and set HF_TOKEN."
            ) from None
        if err.code == 404:
            raise SystemExit(f"{url}: not found. Check the repo name and revision.") from None
        raise


class HttpReader:
    """`read(n)` over HTTP range requests, fetching bigger chunks as the header grows."""

    def __init__(self, url: str, chunk: int = 1 << 20):
        self.url, self.chunk = url, chunk
        self.buf = bytearray()
        self.pos = 0
        self.fetched = 0

    def _fetch(self, need: int) -> None:
        while len(self.buf) < need:
            start = len(self.buf)
            end = start + max(self.chunk, need - start) - 1
            with _request(self.url, {"Range": f"bytes={start}-{end}"}) as resp:
                if resp.status != 206:
                    raise SystemExit(f"{self.url}: the server does not support range requests")
                data = resp.read()
            if not data:
                return
            self.buf += data
            self.fetched += len(data)
            self.chunk = min(self.chunk * 2, 64 << 20)

    def read(self, n: int) -> bytes:
        self._fetch(self.pos + n)
        data = bytes(self.buf[self.pos : self.pos + n])
        self.pos += len(data)
        return data


@dataclass
class ModelFile:
    """One downloadable model: a single .gguf or all parts of a split one."""

    name: str
    paths: list[str]
    size: int
    quant: str
    headers: list[GGUFHeader] = field(default_factory=list)

    @property
    def header(self) -> GGUFHeader:
        return self.headers[0]

    @property
    def tensors(self):
        return [t for h in self.headers for t in h.tensors]


def quant_label(filename: str) -> str:
    matches = _QUANT.findall(Path(filename).name.replace(".gguf", " "))
    return matches[-1].upper().replace("FP16", "F16") if matches else Path(filename).stem


def list_repo(repo: str, revision: str = "main") -> list[tuple[str, int]]:
    """(path, size) of every file in a model repo."""
    url = f"{HF_ENDPOINT}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    files = []
    while url:
        with _request(url) as resp:
            entries = json.load(resp)
            link = resp.headers.get("Link", "")
        files += [(e["path"], (e.get("lfs") or {}).get("size", e.get("size", 0))) for e in entries if e.get("type") == "file"]
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = nxt.group(1) if nxt else None
    return files


def group_models(files: list[tuple[str, int]]) -> list[ModelFile]:
    """Group split parts together and drop projectors and other non-model files."""
    groups: dict[str, ModelFile] = {}
    for path, size in sorted(files):
        name = path.rsplit("/", 1)[-1]
        if "mmproj" in name.lower():
            continue
        m = _SPLIT.match(name) or _PART.match(name)
        if m:
            key = path[: len(path) - len(name)] + m["stem"]
        elif name.endswith(".gguf"):
            key = path
        else:
            continue
        model = groups.setdefault(key, ModelFile(key.rsplit("/", 1)[-1], [], 0, quant_label(key)))
        model.paths.append(path)
        model.size += size
    # some repos hold the same quant twice, e.g. split in two different ways
    unique = {(m.name, round(m.size / 1e7)): m for m in groups.values()}
    return sorted(unique.values(), key=lambda m: m.size)


def smallest_projector(files: list[tuple[str, int]]) -> tuple[str, int] | None:
    """(file name, size) of the smallest vision projector (mmproj) in a repo, if it has one."""
    found = [(size, path.rsplit("/", 1)[-1]) for path, size in files if "mmproj" in path.lower() and path.endswith(".gguf")]
    if not found:
        return None
    size, name = min(found)
    return name, size


def file_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF_ENDPOINT}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"


def load_headers(model: ModelFile, repo: str | None, revision: str = "main") -> ModelFile:
    """Read the header of every split part (a byte-split .partNofM only has one, in part 1)."""
    paths = model.paths
    if _PART.match(paths[0].rsplit("/", 1)[-1]):
        paths = paths[:1]
    for path in paths:
        if repo is None:
            model.headers.append(read_local(path))
        else:
            model.headers.append(read_header(HttpReader(file_url(repo, path, revision)).read))
    return model


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Bytes start..end (inclusive) of a remote file."""
    with _request(url, {"Range": f"bytes={start}-{end}"}) as resp:
        if resp.status != 206:
            raise SystemExit(f"{url}: the server does not support range requests")
        return resp.read()


def _tensor_table_near(url: str, ref: GGUFHeader, slack: int = 256 << 10) -> GGUFHeader | None:
    """Read only the tensor table of a file whose metadata matches `ref` up to a few small keys.

    Quants of one model share the tokenizer and almost all metadata, so their tensor
    tables start within a few kB of the reference's. Finding the first tensor's name
    there and parsing a complete, valid table confirms the position.
    """
    n_tensors, _ = read_counts(fetch_range(url, 0, 23))
    if not ref.tensors or n_tensors == 0:
        return None
    first = ref.tensors[0].name.encode()
    needle = struct.pack("<Q", len(first)) + first
    start = max(0, ref.tensor_info_offset - slack)
    data = fetch_range(url, start, ref.tensor_info_offset + slack + 160 * n_tensors)
    pos = data.find(needle)
    while pos >= 0:
        try:
            tensors = read_tensor_table(data, pos, n_tensors)
        except (GGUFError, struct.error, UnicodeDecodeError):
            pos = data.find(needle, pos + 1)
            continue
        return GGUFHeader(ref.version, ref.metadata, tensors, 0, start + pos)
    return None


def load_repo_headers(models: list[ModelFile], repo: str, revision: str = "main", workers: int = 8) -> None:
    """Fill in the headers of many quants of one model, reading full metadata only once."""
    if not models:
        return
    load_headers(models[0], repo, revision)
    ref = models[0].header

    def one(model: ModelFile) -> None:
        if model.headers:
            return
        if _PART.match(model.paths[0].rsplit("/", 1)[-1]) or len(model.paths) > 1:
            load_headers(model, repo, revision)  # split parts 2.. have their own short headers
            return
        header = _tensor_table_near(file_url(repo, model.paths[0], revision), ref)
        if header is None:
            load_headers(model, repo, revision)
        else:
            model.headers.append(header)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, models[1:]))


def local_model(path: str) -> ModelFile:
    """A local .gguf; finds the other parts of a split file next to it."""
    p = Path(path)
    m = _SPLIT.match(p.name)
    parts = sorted(p.parent.glob(f"{m['stem']}-*-of-{m['count']}.gguf")) if m else [p]
    model = ModelFile(m["stem"] if m else p.name, [str(x) for x in parts], sum(x.stat().st_size for x in parts), quant_label(p.name))
    return load_headers(model, None)


def pick_draft(models: list[ModelFile], quant: str = "") -> ModelFile | None:
    """The quant asked for, or Q8_0: a small draft model loses much of its accuracy at 4 bits, and with it its speed-up.
    Without a Q8_0, the largest file below 16 bits."""
    if quant:
        exact = [m for m in models if m.quant.lower() == quant.lower()]
        return (exact or [m for m in models if quant.lower() in m.name.lower()] or [None])[0]
    q8 = [m for m in models if m.quant == "Q8_0"]
    small = [m for m in models if m.quant not in ("BF16", "F16", "F32")]
    return (q8 or small[-1:] or models[:1] or [None])[0]


def load_draft(ref: str) -> ModelFile:
    """A draft model from a local .gguf, a repo (org/name) or one quant of a repo (org/name:Q4_K_M)."""
    if Path(ref).is_file():
        return local_model(ref)
    repo, _, quant = ref.strip().partition(":")
    if repo.count("/") != 1:
        raise SystemExit(f"--draft {ref!r} is neither a file nor a Hugging Face repo like org/name or org/name:Q8_0")
    model = pick_draft(group_models(list_repo(repo)), quant)
    if model is None:
        raise SystemExit(f"no GGUF files{' matching ' + quant if quant else ''} in {repo}")
    return load_headers(model, repo)
