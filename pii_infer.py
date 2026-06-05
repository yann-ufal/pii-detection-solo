"""
This module is launched once per GCD (1 process per LUMI-G node GCD) by torchrun
and the job spans multiple nodes. Global rank 0 walks the input tree once
and writes a shared file manifest. Every process then reads it,
takes a balanced shard of the list and processes its shard completely independently:
no c10d comms, no torchrun cross-node rendezvous.
Each process computes its global coordinates from Slurm + torchrun environment
variables:

global_rank  = SLURM_NODEID * LOCAL_WORLD_SIZE + LOCAL_RANK
global_world = SLURM_NNODES * LOCAL_WORLD_SIZE

VRAM safety, hardware-measured approach:

At startup each rank runs a short VRAM calibration: it forwards dummy
single sequences along a geometric length ladder that climbs toward the
context window, measures peak allocated memory, and fits
peak = base + b*(N*L) + c*(N*L^2)
From that fit it derives SAFE_SINGLE, the largest single-sequence
length whose predicted peak stays under a working target.
Documents are bucketed onto a fixed geometric ladder of padded sequence lengths
and each bucket runs at a single fixed batch size.
So torch.compile for flex attention never recompiles mid-run.

"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from transformers import pipeline
from transformers.pipelines.token_classification import AggregationStrategy


# Root of the input folder tree. symlinks or files.
INPUT_ROOT = Path("/scratch/project_465002530/users/yannmonn/test/inputs")

# Root of the output folder tree. The subfolder structure of INPUT_ROOT is
# reproduced under OUTPUT_ROOT, same filenames
OUTPUT_ROOT = Path("/scratch/project_465002530/users/yannmonn/test/outputs")

# HuggingFace model id
MODEL_ID = "openai/privacy-filter"

# Group sub-word tokens into whole-word entity spans
AGGREGATION = "first"
AGG_ENUM = AggregationStrategy(AGGREGATION)

# Attention kernel. flex is th eonly performant linear space complexity option on rocm
# if flex_attention does not load, the rank raises and the job fails loudly
ATTN_IMPLEMENTATIONS = ("flex_attention",)

# bf16: halves weight + activation memory vs fp32 and is the native compute dtype on MI250X
DTYPE = torch.bfloat16

# context limit of the model, also the absolute truncation cap
MODEL_MAX_TOKENS = 128_000

# VRAM safety budget per GCD. The card has 64 GB
VRAM_CEILING_GB = 50.0
VRAM_TARGET_FRACTION = 0.95

# candidate documents to buffer before length-sorting + batching. goes to cpu RAM
BUFFER_DOCS = 4096

# cap on sequences per forward pass to bound per-sequence overhead for batches of tiny docs
MAX_BATCH_SEQS = 512

# flex attention runs under torch.compile, which compiles a fresh kernel per
# distinct (batch_size, seq_len). To stop the wildly varying document lengths from
# triggering a recompile storm by bucketing geometrically.
# MIN_SEQ_BUCKET is the smallest pad length then the ladder climbs by SEQ_BUCKET_FACTOR
# up to the calibrated safe-single length.
MIN_SEQ_BUCKET = 256
SEQ_BUCKET_FACTOR = 2

# torch.compile's per-callable shape cache defaults to 8 entries
# with more buckets than that, dynamo stops compiling new shapes and drops
# to qudratic eager attention (BAD!)
import torch._dynamo  # noqa: E402
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 256

ZSTD_LEVEL = 3
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
MANIFEST_PATH = OUTPUT_ROOT / f"_file_manifest.{JOB_ID}.tsv"

# 8 GCDs map to NUMA-local CPU CCDs: pinning each rank's tokenisation/zstd threads
# to its GCD-local cores avoids cross-NUMA memory traffic
LUMI_GCD_CPU_CORES = {
    0: range(49, 56), 1: range(57, 64), 2: range(17, 24), 3: range(25, 32),
    4: range(1, 8),   5: range(9, 16),  6: range(33, 40), 7: range(41, 48),
}



def dist_env() -> tuple[int, int, int, int]:
    """
    Compute this process's GLOBAL rank/world from Slurm + torchrun env vars.
    torchrun (ignited standalone) sets LOCAL_RANK and LOCAL_WORLD_SIZE
    Slurm sets SLURM_NODEID and SLURM_NNODES

    Returns ``(global_rank, global_world_size, local_rank, local_world_size)``
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    n_nodes = int(os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    rank = node_id * local_world + local_rank
    world_size = n_nodes * local_world
    return rank, world_size, local_rank, local_world


def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def file_size(path: Path) -> int:
    """size of the file a symlink points at"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def discover_files(root: Path) -> list[Path]:

    files: list[Path] = []
    # Match both the .zst and .zstd spellings of the zstd extension. pathlib
    # anchors the glob to the end of the name, so "*.jsonl.zst" does NOT match a
    # "*.jsonl.zstd" file -- the two patterns are disjoint, no dedup needed.
    for pattern in ("*.jsonl.zst", "*.jsonl.zstd"):
        for p in root.rglob(pattern):
            try:
                if p.is_file():
                    files.append(p)
            except OSError:
                continue  # broken symlink
    return sorted(files)


def get_manifest(root: Path, rank: int, timeout: float = 1800.0) -> list[tuple[Path, int]]:
    """
    Return ``[(path, size), ...]`` for every input file
    """
    if rank == 0 and not MANIFEST_PATH.exists():
        entries = [(p, file_size(p)) for p in discover_files(root)]
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for p, sz in entries:
                fh.write(f"{sz}\t{p}\n")
        os.replace(tmp, MANIFEST_PATH)

    waited = 0.0
    while not MANIFEST_PATH.exists():
        if waited >= timeout:
            raise RuntimeError(f"manifest {MANIFEST_PATH} did not appear within {timeout:.0f}s")
        time.sleep(2.0)
        waited += 2.0

    entries: list[tuple[Path, int]] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sz, p = line.split("\t", 1)
            entries.append((Path(p), int(sz)))
    return entries


def shard_for_rank(entries: list[tuple[Path, int]], rank: int, world_size: int) -> list[Path]:
    """
    Assign files to ranks with greedy longest-processing-time balancing, using
    the sizes already carried in the manifest (no re-stat).
    Every rank runs this identical, deterministic computation and keeps only its own files
    — no comms
    """
    if world_size <= 1:
        return sorted(p for p, _ in entries)

    sized = sorted(entries, key=lambda e: e[1], reverse=True)
    loads = [0] * world_size
    buckets: list[list[Path]] = [[] for _ in range(world_size)]
    for path, size in sized:
        target = min(range(world_size), key=lambda r: loads[r])
        buckets[target].append(path)
        loads[target] += size

    # sort this rank's own files by name for stable, readable progress logs
    return sorted(buckets[rank])


def iter_rows(path: Path):
    """Yield each decoded JSON object from a *.jsonl.zst file"""
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # one bad line does not halt the whole file
                continue


def should_process(row: dict) -> bool:
    """
    Decide whether a row's text must be run through the model

    Skip the row only when ``propella-4b.pii_presence == "no_pii"``. If the
    ``propella-4b`` key is missing entirely, go ahead.
    """
    prop = row.get("propella-4b")
    if prop is None:
        return True
    if isinstance(prop, str):
        try:
            prop = json.loads(prop)
        except json.JSONDecodeError:
            return True
    if isinstance(prop, dict) and prop.get("pii_presence") == "no_pii":
        return False
    return True


def correct_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    """
    Trim whitespace off a model span and return ``(start, end, value)``

    Deals with the GPT-style (byte-level BPE) tokenizer that folds the
    whitespace preceding a word into that word's first token

    ``end`` is exclusive: ``value == text[start:end]``
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, text[start:end]


def out_path_for(in_path: Path) -> Path:
    """Mirror in_path under OUTPUT_ROOT, preserving the relative tree"""
    rel = in_path.relative_to(INPUT_ROOT)
    return OUTPUT_ROOT / rel


def entities_to_rows(doc_id: str, text: str, entities: list[dict]):
    """Turn one document's model output into output JSONL rows"""
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        start, end, value = correct_span(text, int(start), int(end))
        if not value:   # empty after trimming whitespace
            continue
        name = ent.get("entity_group") or ent.get("entity")
        yield {
            "id": doc_id,
            "name": name,
            "value": value,
            "start_pos": start,
            "end_pos": end,
        }


def calibrate_vram(model, device, rank: int, ceiling_bytes: float, target_bytes: float):
    """
    Measure how peak GPU memory grows with batch size and sequence length

    return ``(predict, safe_single)`` where:
    predict(N, L) estimates peak bytes for a forward pass of N sequences each padded to L tokens, and
    safe_single is the largest single-sequence length whose predicted peak stays under target_bytes

    fit peak = base + b*(N*L) + c*(N*L^2)
    if the banded kernel plays nice, c is near 0 and safe_single lands at MODEL_MAX_TOKENS
    """
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    vocab = int(getattr(model.config, "vocab_size", 32000))

    rows: list[list[int]] = []   # [N*L, N*L^2] per probe
    peaks: list[float] = []      # measured peak bytes above base

    def probe(length: int):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ids = torch.randint(0, vocab, (1, length), device=device, dtype=torch.long)
        mask = torch.ones((1, length), device=device, dtype=torch.long)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize(device)
        rows.append([length, length * length])
        peaks.append(float(torch.cuda.max_memory_allocated(device) - base))
        del ids, mask

    def fit() -> tuple[float, float]:
        A = np.asarray(rows, dtype=np.float64)
        y = np.asarray(peaks, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return max(float(coef[0]), 0.0), max(float(coef[1]), 0.0)

    # Two seeds fit the 2 parameters climbing a geometric ladder
    # continues only while the current fit predicts it stays under the ceiling
    probe(512)
    probe(1024)
    L = 1024
    while L < MODEL_MAX_TOKENS:
        b, c = fit()
        nxt = min(L * 2, MODEL_MAX_TOKENS)
        if base + b * nxt + c * nxt * nxt > ceiling_bytes:
            break
        probe(nxt)
        L = nxt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    b, c = fit()

    def predict(n: int, length: int) -> float:
        return base + b * (n * length) + c * (n * length * length)

    rhs = target_bytes - base
    if c > 0:
        length = (-b + math.sqrt(b * b + 4 * c * rhs)) / (2 * c)
    elif b > 0:
        length = rhs / b
    else:
        length = float(MODEL_MAX_TOKENS)
    safe_single = int(max(1024, min(MODEL_MAX_TOKENS, length)))

    quad = c * safe_single * safe_single
    lin = b * safe_single
    regime = "quadratic" if quad > lin else "linear"
    log(rank,
        f"VRAM calib: base={base / 2**30:.2f}GiB "
        f"target={target_bytes / 2**30:.1f}GiB ceiling={ceiling_bytes / 2**30:.0f}GiB "
        f"probes={[r[0] for r in rows]} "
        f"b={b:.1f} B/tok c={c:.3e} B/tok^2 attn~{regime} "
        f"safe_single_tokens={safe_single} "
        f"(predicted peak @safe_single={predict(1, safe_single) / 2**30:.1f}GiB)")
    return predict, safe_single


def build_seq_buckets(safe_single: int) -> list[int]:
    """
    Fixed geometric ladder of padded sequence lengths, top rung == ``safe_single``
    (the truncation cap). Every document is padded UP to its bucket, so the model
    only ever sees these few lengths and flex_attention's compiled kernel is reused
    rather than recompiled per distinct length.
    """
    buckets: list[int] = []
    L = MIN_SEQ_BUCKET
    while L < safe_single:
        buckets.append(L)
        L *= SEQ_BUCKET_FACTOR
    buckets.append(safe_single)
    return buckets


def bucket_for(length: int, buckets: list[int]) -> int:
    """Index of the smallest bucket >= ``length``. Documents are truncated to the
    top bucket, so the loop always finds one."""
    for bi, L in enumerate(buckets):
        if length <= L:
            return bi
    return len(buckets) - 1


def warmup_buckets(clf, predict, target_bytes, safe_single, rank):
    """
    Pick one fixed batch size per sequence bucket from the memory fit, then run a
    dummy forward at each (batch_size, seq_len) shape so flex_attention compiles
    every shape ONCE, up front -- the steady state then has zero recompiles.

    The dummy forward doubles as an OOM safety check: if a bucket's chosen batch
    size does not actually fit, it is halved until it does, so no real batch can
    OOM later (and OOM-driven reshaping, which would recompile, never happens).

    Returns ``(buckets, batch_sizes)``.
    """
    buckets = build_seq_buckets(safe_single)
    pad_id = clf.tokenizer.pad_token_id if clf.tokenizer.pad_token_id is not None else 0

    def fit_bsz(L: int) -> int:
        n = 1
        while n < MAX_BATCH_SEQS and predict(n + 1, L) <= target_bytes:
            n += 1
        return n

    batch_sizes = [fit_bsz(L) for L in buckets]

    if not torch.cuda.is_available():
        return buckets, batch_sizes

    t0 = time.time()
    for bi, L in enumerate(buckets):
        while True:
            B = batch_sizes[bi]
            ids = torch.full((B, L), pad_id, dtype=torch.long, device=clf.device)
            mask = torch.zeros((B, L), dtype=torch.long, device=clf.device)
            mask[:, 0] = 1
            try:
                with torch.no_grad():
                    clf.model(input_ids=ids, attention_mask=mask)
                del ids, mask
                break
            except torch.cuda.OutOfMemoryError:
                del ids, mask
                torch.cuda.empty_cache()
                if B == 1:
                    log(rank, f"warmup: bucket L={L} OOMs even at batch=1")
                    break
                batch_sizes[bi] = max(1, B // 2)
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(clf.device)
    log(rank, f"flex warmup: compiled {len(buckets)} shapes in {time.time() - t0:.1f}s "
              f"buckets={buckets} batch_sizes={batch_sizes}")
    return buckets, batch_sizes


def process_file(in_path: Path, clf, buckets, batch_sizes, safe_single, rank: int):
    """
    Run inference over one input file and write output.

    Documents are streamed, tokenised in read-sized chunks, and routed into a
    per-bucket pending queue. A bucket is flushed as a full fixed-shape batch as
    soon as it fills.
    """
    final_out = out_path_for(in_path)
    if final_out.exists():
        log(rank, f"skip (already done): {in_path.name}")
        return

    final_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = final_out.with_suffix(final_out.suffix + ".tmp")

    tok = clf.tokenizer
    device = clf.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)

    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    n_rows = n_entities = n_trunc = 0
    t0 = time.time()

    with tmp_out.open("wb") as raw, cctx.stream_writer(raw) as writer:
        out = io.TextIOWrapper(writer, encoding="utf-8")
        pending: list[list[dict]] = [[] for _ in buckets]

        def emit(doc_id, text, ents):
            nonlocal n_entities
            for r in entities_to_rows(doc_id, text, ents):
                out.write(json.dumps(r, ensure_ascii=False))
                out.write("\n")
                n_entities += 1

        def flush_chunk(bi: int, chunk: list[dict]):
            # chunk holds <= batch_sizes[bi] real docs. the batch dim is padded out
            # to the fixed size with masked dummy rows to keep the shape constant.
            L = buckets[bi]
            B = batch_sizes[bi]
            k = len(chunk)
            ids_t = torch.full((B, L), pad_id, dtype=torch.long)
            mask_t = torch.zeros((B, L), dtype=torch.long)
            for j, it in enumerate(chunk):
                li = it["len"]
                ids_t[j, :li] = torch.as_tensor(it["ids"], dtype=torch.long)
                mask_t[j, :li] = 1
            for j in range(k, B):       # dummy rows: one live token avoids empty softmax
                mask_t[j, 0] = 1
            with torch.no_grad():
                logits = clf.model(input_ids=ids_t.to(device),
                                   attention_mask=mask_t.to(device)).logits
            # fp32 on host: postprocess runs a numpy softmax, which cannot take bf16
            logits = logits.to(dtype=torch.float32, device="cpu")
            for j, it in enumerate(chunk):
                li = it["len"]
                model_outputs = {
                    "logits": logits[j:j + 1, :li, :],
                    "input_ids": torch.as_tensor([it["ids"]], dtype=torch.long),
                    "offset_mapping": torch.as_tensor([it["offs"]], dtype=torch.long),
                    "special_tokens_mask": torch.as_tensor([it["stm"]], dtype=torch.long),
                    "sentence": it["text"],
                }
                ents = clf.postprocess([model_outputs], aggregation_strategy=AGG_ENUM)
                emit(it["doc_id"], it["text"], ents)
            del ids_t, mask_t, logits

        def add(doc_id, text, ids, offs, stm):
            bi = bucket_for(len(ids), buckets)
            pending[bi].append({"doc_id": doc_id, "text": text, "ids": ids,
                                "offs": offs, "stm": stm, "len": len(ids)})
            B = batch_sizes[bi]
            if len(pending[bi]) >= B:
                flush_chunk(bi, pending[bi][:B])
                del pending[bi][:B]

        read_buf: list[tuple[str, str]] = []

        def tokenise_and_route():
            nonlocal n_trunc
            if not read_buf:
                return
            texts = [t for _, t in read_buf]
            enc = tok(texts, add_special_tokens=True, return_offsets_mapping=True,
                      return_special_tokens_mask=True, truncation=False)
            for i, (doc_id, text) in enumerate(read_buf):
                ids = enc["input_ids"][i]
                offs = enc["offset_mapping"][i]
                stm = enc["special_tokens_mask"][i]
                if len(ids) > safe_single:
                    re = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                             return_special_tokens_mask=True, truncation=True,
                             max_length=safe_single)
                    ids, offs, stm = re["input_ids"], re["offset_mapping"], re["special_tokens_mask"]
                    n_trunc += 1
                add(doc_id, text, ids, offs, stm)
            read_buf.clear()

        for row in iter_rows(in_path):
            n_rows += 1
            if not should_process(row):
                continue
            text = row.get("text")
            if not text or not isinstance(text, str):
                continue
            read_buf.append((str(row.get("id", "")), text))
            if len(read_buf) >= BUFFER_DOCS:
                tokenise_and_route()
        tokenise_and_route()
        for bi in range(len(buckets)):      # EOF: flush every bucket's remainder
            if pending[bi]:
                flush_chunk(bi, pending[bi])
                pending[bi] = []
        out.flush()

    os.replace(tmp_out, final_out)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if on_gpu else 0.0
    extra = f", {n_trunc} truncated@{safe_single}tok" if n_trunc else ""
    log(rank,
        f"done {in_path.name}: {n_rows} rows -> {n_entities} entities "
        f"in {dt:.1f}s (peak {peak:.1f}GB{extra})")
    if peak > VRAM_CEILING_GB:
        log(rank, f"WARNING: peak {peak:.1f}GB exceeded the {VRAM_CEILING_GB}GB ceiling")



def main():
    rank, world_size, local_rank, local_world = dist_env()

    if not torch.cuda.is_available():
        log(rank, "WARNING: no GPU visible -running on CPU")
        device = -1
    else:
        torch.cuda.set_device(local_rank)
        device = local_rank

    # Pin this rank to the CPU cores in the same NUMA domain as its GCD,
    # so tokenisation + zstd threads don't pay cross-NUMA memory-bandwidth taxes
    bound_cores = None
    if hasattr(os, "sched_setaffinity") and local_world == 8 and local_rank in LUMI_GCD_CPU_CORES:
        try:
            allowed = os.sched_getaffinity(0)
            target = {c for c in LUMI_GCD_CPU_CORES[local_rank] if c in allowed}
            if target:
                os.sched_setaffinity(0, target)
                bound_cores = len(target)
        except OSError:
            bound_cores = None

    if bound_cores:
        torch.set_num_threads(bound_cores)
    else:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or local_world)
        torch.set_num_threads(max(1, cores // max(1, local_world)))

    log(rank, f"world_size={world_size} node_local_rank={local_rank} "
              f"local_world={local_world} device={device} "
              f"cpu_threads={torch.get_num_threads()}"
              f"{' (GCD-NUMA pinned)' if bound_cores else ''}")

    pipe_kwargs = dict(
        task="token-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        aggregation_strategy=AGGREGATION,
        device=device,
        dtype=DTYPE,
    )
    clf = None
    for attn_impl in ATTN_IMPLEMENTATIONS:
        try:
            clf = pipeline(**pipe_kwargs,
                           model_kwargs={"attn_implementation": attn_impl})
            break
        except (ValueError, ImportError, RuntimeError) as exc:
            log(rank, f"attn_implementation={attn_impl!r} unavailable ({exc!r})")
    if clf is None:
        raise RuntimeError(
            f"none of {ATTN_IMPLEMENTATIONS} could be loaded for {MODEL_ID}; "
            f"refusing to run on eager attention (O(N^2) memory, too slow for "
            f"this workload). Diagnose with attn_probe.py on a login node.")
    if not clf.tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset mappings")
    clf.tokenizer.model_max_length = MODEL_MAX_TOKENS
    attn = getattr(clf.model.config, "_attn_implementation", "?")
    log(rank, f"loaded {MODEL_ID} dtype={DTYPE} attn_impl={attn}")

    if device == -1:
        # CPU: cap batch by MAX_BATCH_SEQS
        def predict(n, length):
            return 0.0
        target_bytes = float("inf")
        safe_single = MODEL_MAX_TOKENS
    else:
        ceiling_bytes = VRAM_CEILING_GB * 2**30
        target_bytes = ceiling_bytes * VRAM_TARGET_FRACTION
        predict, safe_single = calibrate_vram(
            clf.model, clf.device, rank, ceiling_bytes, target_bytes)

    buckets, batch_sizes = warmup_buckets(clf, predict, target_bytes, safe_single, rank)

    entries = get_manifest(INPUT_ROOT, rank)
    if rank == 0:
        log(rank, f"discovered {len(entries)} input files under {INPUT_ROOT}")
    if not entries:
        log(rank, "no input files found; nothing to do")
        return

    my_files = shard_for_rank(entries, rank, world_size)
    log(rank, f"assigned {len(my_files)} files")

    try:
        for i, in_path in enumerate(my_files, 1):
            try:
                process_file(in_path, clf, buckets, batch_sizes, safe_single, rank)
            except Exception as exc:  # keep going on the rest of the shard
                log(rank, f"ERROR on {in_path}: {exc!r}")
            if i % 10 == 0:
                log(rank, f"progress {i}/{len(my_files)} files")
        log(rank, "shard complete")
    finally:
        if rank == 0:
            try:
                MANIFEST_PATH.unlink(missing_ok=True)
                log(rank, f"removed manifest {MANIFEST_PATH.name}")
            except OSError as exc:
                log(rank, f"could not remove manifest {MANIFEST_PATH.name}: {exc!r}")


if __name__ == "__main__":
    sys.exit(main())
