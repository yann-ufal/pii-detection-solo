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
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd
from transformers import pipeline


# Root of the input folder tree. symlinks or files.
INPUT_ROOT = Path("/scratch/project_465002530/users/yannmonn/test/inputs")

# Root of the output folder tree. The subfolder structure of INPUT_ROOT is
# reproduced under OUTPUT_ROOT, same filenames
OUTPUT_ROOT = Path("/scratch/project_465002530/users/yannmonn/test/outputs")

# HuggingFace model id
MODEL_ID = "openai/privacy-filter"

# Group sub-word tokens into whole-word entity spans
AGGREGATION = "first"
assert AGGREGATION == "first", "aggregate_first only implements AGGREGATION='first'"

# Attention kernel. flex is th eonly performant linear space complexity option on rocm
# if flex_attention does not load, the rank raises and the job fails loudly
ATTN_IMPLEMENTATIONS = ("flex_attention",)

# bf16: halves weight + activation memory vs fp32 and is the native compute dtype on MI250X
DTYPE = torch.bfloat16

# context limit of the model, also the absolute truncation cap
MODEL_MAX_TOKENS = 128000

# per GCD
VRAM_CEILING_GB = 64
VRAM_TARGET_FRACTION = 0.80

# this is the tok() buckets granularity at which tokenised batches reach the GPU queue
BUFFER_DOCS = 2048

# cap on sequences per forward pass to bound per-sequence overhead for batches of tiny docs
MAX_BATCH_SEQS = 2048

# This CPU Python thread holds the GIL like dear Life
N_POSTPROCESS_WORKERS = 1 # NO. STOP. DO NOT INCREMENT. POISON.
GPU_QUEUE_DEPTH = 4
POST_QUEUE_DEPTH = 8     # forwarded logits (on host) waiting for postprocess
WRITE_QUEUE_DEPTH = 64   # text blocks waiting to be written

# queue sentinels: _DONE = clean end-of-stream, _ABORT = a stage failed, unwind
_DONE = object()
_ABORT = object()

# flex attention runs under torch.compile, which compiles a fresh kernel per distinct
# (batch_size, seq_len) at launch, then nothing ever again.
# MIN_SEQ_BUCKET is the smallest pad length then the ladder climbs by
# SEQ_BUCKET_FACTOR up to the calibrated safe-single length.
MIN_SEQ_BUCKET = 256
SEQ_BUCKET_FACTOR = 1.5

# torch.compile's per-callable shape cache defaults to 8 entries
# with more buckets than that, dynamo stops compiling new shapes and drops
# to qudratic eager attention (BAD!)
import torch._dynamo
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
    for pattern in ("*.jsonl.zst", "*.jsonl.zstd", "*.jsonl.gz"):
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


def extract_doc_id(row: dict) -> str:
    """
    Pull the document id from a row.

    Files differ in where the id lives; it is always exactly one of these.
    Probe them in order and use the first that is present:
      1. top-level "id"
      2. "metadata" -> "WARC-Record-ID"  (nested one level)
      3. top-level "warc_record_id"
      4. top-level "Document ID"
      5. top-level "uuid"
    """
    val = row.get("id")
    if val is not None:
        return str(val)

    meta = row.get("metadata")
    if isinstance(meta, dict):
        val = meta.get("WARC-Record-ID")
        if val is not None:
            return str(val)

    for key in ("warc_record_id", "Document ID", "uuid"):
        val = row.get(key)
        if val is not None:
            return str(val)

    return ""


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


def build_label_meta(clf) -> tuple:
    """
    Precompute the per-label lookup tables aggregate_first needs, once per process
    This model tags in the BIOES scheme B-/I-/E-/S- per entity type
    HF's get_tag()/group_sub_entities only understand B-/I-

    Returns (name, group_tag, opens, closes, unk_id) indexed by label id
    """
    raw = clf.model.config.id2label
    n = clf.model.config.num_labels
    labels = [raw[i] if i in raw else raw[str(i)] for i in range(n)]

    def strip(l):  # drop a leading B-/I-/E-/S- prefix. "O" unchanged
        return l[2:] if l[:2] in ("B-", "I-", "E-", "S-") else l

    return (
        np.array([strip(l) for l in labels], dtype=object),
        np.array([strip(l) for l in labels], dtype=object),
        np.array([l[:2] in ("B-", "S-") for l in labels], dtype=bool),
        np.array([l[:2] in ("E-", "S-") for l in labels], dtype=bool),
        clf.tokenizer.unk_token_id,
    )


# all Unicode whitespace code points. every whitespace char is <= U+3000
WHITESPACE_CPS = np.array(sorted(c for c in range(0x3001) if chr(c).isspace()), dtype=np.uint32)

def aggregate_first(logits_np, input_ids, offsets, special_mask, sentence, meta):
    """
    Vectorised equivalent of ``TokenClassificationPipeline.postprocess`` for
    ``aggregation_strategy="first"`` with the default ``ignore_labels=["O"]`` and
    the fallback (non-word-aware) ``is_subword`` heuristic: the path this model
    takes (that "Tokenizer does not support real words" bla-bla nonsense)

    only ``(entity_group, start, end)`` are produced. entities_to_rows needs
    nothing else: the per-token softmax, word strings and mean scores HF
    computes are all skipped

    Two modifs: word boundaries use all unicode whitespace, and grouping is BIOES-aware

    ``logits_np`` is the (li, C) fp32 slice for one document.
    """
    name_arr, gtag_arr, opens_arr, closes_arr, unk_id = meta

    special = np.asarray(special_mask, dtype=bool)
    keep = ~special
    if not keep.any():
        return []
    idx = np.nonzero(keep)[0]   # non-special token positions

    pred_k = logits_np.argmax(-1)[idx]  # per-token label id (first token wins later)
    offs = np.asarray(offsets)
    starts = offs[idx, 0].astype(np.int64)
    ends = offs[idx, 1].astype(np.int64)
    ids_k = np.asarray(input_ids)[idx]

    if sentence:
        cps = np.frombuffer(sentence.encode("utf-32-le"), dtype=np.uint32)
    else:
        cps = np.empty(0, dtype=np.uint32)
    n = cps.shape[0]
    is_space = np.isin(cps, WHITESPACE_CPS)
    prev_sp = np.zeros(starts.shape, dtype=bool)
    cur_sp = np.zeros(starts.shape, dtype=bool)
    sm1 = starts - 1
    m = (sm1 >= 0) & (sm1 < n)
    prev_sp[m] = is_space[sm1[m]]
    m2 = (starts >= 0) & (starts < n)
    cur_sp[m2] = is_space[starts[m2]]
    is_subword = (starts > 0) & ~(prev_sp | cur_sp)
    if unk_id is not None:  # HF forces is_subword False on <unk>
        is_subword &= ids_k != unk_id

    # words: a new word starts at the first kept token and at every non-subword token
    # the word's label is its FIRST token's. its span is first.start..last.end.
    K = idx.shape[0]
    word_start = ~is_subword
    word_start[0] = True
    w = np.nonzero(word_start)[0]
    word_pred = pred_k[w]
    word_start_off = starts[w]
    word_last = np.empty_like(w)
    word_last[:-1] = w[1:] - 1
    word_last[-1] = K - 1
    word_end_off = ends[word_last]

    # BIOES entity groups instead of HF's BIO-only rule
    W = w.shape[0]
    opens = opens_arr[word_pred]
    closes = closes_arr[word_pred]
    new_group = np.ones(W, dtype=bool)
    if W > 1:
        gtag = gtag_arr[word_pred]
        new_group[1:] = (gtag[1:] != gtag[:-1]) | opens[1:] | closes[:-1]
    g = np.nonzero(new_group)[0]
    g_name = name_arr[word_pred[g]]
    g_start = word_start_off[g]
    g_last = np.empty_like(g)
    g_last[:-1] = g[1:] - 1
    g_last[-1] = W - 1
    g_end = word_end_off[g_last]

    sel = g_name != "O" # ignore_labels == ["O"]
    return [{"entity_group": nm, "start": int(s), "end": int(e)}
            for nm, s, e in zip(g_name[sel], g_start[sel], g_end[sel])]


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
    Fixed geometric ladder of padded sequence lengths up to ``safe_single``
    bucket construsction for flex_attention's compiled kernels recycling
    """
    buckets: list[int] = []
    L = MIN_SEQ_BUCKET
    while L < safe_single:
        buckets.append(L)
        L *= SEQ_BUCKET_FACTOR
    buckets.append(safe_single)
    return buckets


def bucket_for(length: int, buckets: list[int]) -> int:
    """Index of the smallest bucket >= length"""
    for bi, L in enumerate(buckets):
        if length <= L:
            return bi
    return len(buckets) - 1


def warmup_buckets(clf, predict, target_bytes, safe_single, rank):
    """
    Pick one fixed batch size per sequence bucket from the memory fit, then run a
    dummy forward at each (batch_size, seq_len) shape so flex_attention compiles
    every shape once, up front. the steady state then has zero recompiles.

    Returns ``(buckets, batch_sizes)``
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
    four-stage pipeline of threads joined by bounded queues

    [T] producer (1 thread): iter_rows -> tokenise in BUFFER_DOCS chunks ->
        route into per-bucket pending lists -> assemble fixed-shape (B,L) CPU
        tensor batches (dummy-padded) -> gpu_q
    [G] GPU forward (THIS/main thread, the only thread touching the device):
        gpu_q -> H2D + model forward + logits->host(fp32) -> post_q
    [P] postprocess pool (N_POSTPROCESS_WORKERS threads): post_q -> per-doc
        clf.postprocess numpy-softmax -> entity rows as one text block -> write_q
    [W] writer (1 thread): write_q -> zstd stream. Output row order is the batch
        completion order

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
    label_meta = build_label_meta(clf)   # per-label lookup tables for aggregate_first

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)

    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    t0 = time.time()

    gpu_q: queue.Queue = queue.Queue(maxsize=GPU_QUEUE_DEPTH)
    post_q: queue.Queue = queue.Queue(maxsize=POST_QUEUE_DEPTH)
    write_q: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_DEPTH)

    abort = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    # n_rows/n_trunc are touched only by the producer thread. n_entities only by the writer thread
    # each is single-writer: no locking is needed
    stats = {"n_rows": 0, "n_trunc": 0, "n_entities": 0, "n_tokens": 0}  # TEMP: n_tokens for throughput

    # Each thread accumulates wall-seconds in a LOCAL dict and merges it once,
    # on exit, so the hot path stays lock-free. Postprocess keys are summed
    # across all workers
    #   gpu_wait_in  high -> producer/tokenise starves the GPU
    #   gpu_wait_out high -> postprocess too slow, GPU blocked on backpressure
    timers: dict[str, float] = {}
    timers_lock = threading.Lock()
    pc = time.perf_counter

    def add_times(d: dict):
        with timers_lock:
            for k, v in d.items():
                timers[k] = timers.get(k, 0.0) + v

    def fail(stage: str, exc: BaseException):
        errors.append((stage, exc))
        abort.set()

    def q_put(q: queue.Queue, item) -> bool:
        # abort-aware put: a dead downstream must not deadlock an upstream on a full queue
        # returns False once abort is set so the caller can unwind
        while not abort.is_set():
            try:
                q.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def q_get(q: queue.Queue):
        # abort-aware get: returns _ABORT once abort is set so loops can exit
        while not abort.is_set():
            try:
                return q.get(timeout=0.25)
            except queue.Empty:
                continue
        return _ABORT

    # [T]
    def producer():
        tm = {"prod_read": 0.0, "prod_tok": 0.0, "prod_makejob": 0.0,
              "prod_gpuq_put_wait": 0.0, "prod_jobs": 0.0}
        try:
            pending: list[list[dict]] = [[] for _ in buckets]

            def make_job(bi: int, chunk: list[dict]):
                # the batch dim is padded to the fixed size with masked dummy rows
                L = buckets[bi]
                B = batch_sizes[bi]
                ids_t = torch.full((B, L), pad_id, dtype=torch.long)
                mask_t = torch.zeros((B, L), dtype=torch.long)
                for j, it in enumerate(chunk):
                    li = it["len"]
                    ids_t[j, :li] = torch.as_tensor(it["ids"], dtype=torch.long)
                    mask_t[j, :li] = 1
                for j in range(len(chunk), B):  # dummy rows: 1 token avoids empty softmax
                    mask_t[j, 0] = 1
                return (ids_t, mask_t, chunk)

            def flush_job(bi: int, chunk: list[dict]):
                s = pc()
                job = make_job(bi, chunk)
                tm["prod_makejob"] += pc() - s
                s = pc()
                ok = q_put(gpu_q, job)
                tm["prod_gpuq_put_wait"] += pc() - s
                tm["prod_jobs"] += 1
                return ok

            def route(doc_id, text, ids, offs, stm):
                stats["n_tokens"] += len(ids)  # TEMP: real (post-truncation) tokens fed to GPU
                bi = bucket_for(len(ids), buckets)
                pending[bi].append({"doc_id": doc_id, "text": text, "ids": ids,
                                    "offs": offs, "stm": stm, "len": len(ids)})
                if len(pending[bi]) >= batch_sizes[bi]:
                    B = batch_sizes[bi]
                    flush_job(bi, pending[bi][:B])
                    del pending[bi][:B]

            read_buf: list[tuple[str, str]] = []

            def tokenise_and_route():
                if not read_buf:
                    return
                texts = [txt for _, txt in read_buf]
                s = pc()
                enc = tok(texts, add_special_tokens=True, return_offsets_mapping=True,
                          return_special_tokens_mask=True, truncation=False)
                tm["prod_tok"] += pc() - s
                for i, (doc_id, text) in enumerate(read_buf):
                    ids = enc["input_ids"][i]
                    offs = enc["offset_mapping"][i]
                    stm = enc["special_tokens_mask"][i]
                    if len(ids) > safe_single:
                        s = pc()
                        re = tok(text, add_special_tokens=True, return_offsets_mapping=True,
                                 return_special_tokens_mask=True, truncation=True,
                                 max_length=safe_single)
                        tm["prod_tok"] += pc() - s
                        ids, offs, stm = re["input_ids"], re["offset_mapping"], re["special_tokens_mask"]
                        stats["n_trunc"] += 1
                    route(doc_id, text, ids, offs, stm)
                read_buf.clear()

            rows = iter_rows(in_path)
            while True:
                if abort.is_set():
                    return
                s = pc()
                try:
                    row = next(rows)
                except StopIteration:
                    break
                tm["prod_read"] += pc() - s
                stats["n_rows"] += 1
                if not should_process(row):
                    continue
                text = row.get("text")
                if not text or not isinstance(text, str):
                    continue
                read_buf.append((extract_doc_id(row), text))
                if len(read_buf) >= BUFFER_DOCS:
                    tokenise_and_route()
            tokenise_and_route()
            for bi in range(len(buckets)):  # EOF: flush every bucket's remainder
                if pending[bi]:
                    flush_job(bi, pending[bi])
                    pending[bi] = []
        except Exception as exc:
            fail("tokenize", exc)
        finally:
            add_times(tm)
            if not abort.is_set():
                gpu_q.put(_DONE)    # single sentinel ends the GPU stage

    # [P]
    def postprocessor():
        # summed across all N_POSTPROCESS_WORKERS workers
        tm = {"post_q_get_wait": 0.0, "post_proc": 0.0, "post_writeq_put_wait": 0.0}
        try:
            while True:
                s = pc()
                item = q_get(post_q)
                tm["post_q_get_wait"] += pc() - s
                if item is _ABORT or item is _DONE:
                    return
                logits, chunk = item
                s = pc()
                parts: list[str] = []
                cnt = 0
                logits_np = logits.numpy()   # (B, L, C) fp32, zero-copy view of the host tensor
                for j, it in enumerate(chunk):
                    li = it["len"]
                    ents = aggregate_first(logits_np[j, :li], it["ids"], it["offs"],
                                           it["stm"], it["text"], label_meta)
                    for r in entities_to_rows(it["doc_id"], it["text"], ents):
                        parts.append(json.dumps(r, ensure_ascii=False))
                        cnt += 1
                tm["post_proc"] += pc() - s
                if parts:
                    s = pc()
                    ok = q_put(write_q, ("\n".join(parts) + "\n", cnt))
                    tm["post_writeq_put_wait"] += pc() - s
                    if not ok:
                        return
        except Exception as exc:
            fail("postprocess", exc)
        finally:
            add_times(tm)

    # [W]
    def make_writer(out):
        def writer():
            tm = {"write_q_get_wait": 0.0, "write_io": 0.0}
            try:
                while True:
                    s = pc()
                    item = q_get(write_q)
                    tm["write_q_get_wait"] += pc() - s
                    if item is _ABORT or item is _DONE:
                        return
                    block, cnt = item
                    s = pc()
                    out.write(block)
                    tm["write_io"] += pc() - s
                    stats["n_entities"] += cnt
            except Exception as exc:
                fail("write", exc)
            finally:
                add_times(tm)
        return writer

    with tmp_out.open("wb") as raw, cctx.stream_writer(raw) as cwriter:
        out = io.TextIOWrapper(cwriter, encoding="utf-8")

        prod_t = threading.Thread(target=producer, name="tok", daemon=True)
        post_ts = [threading.Thread(target=postprocessor, name=f"post{i}", daemon=True)
                   for i in range(N_POSTPROCESS_WORKERS)]
        write_t = threading.Thread(target=make_writer(out), name="write", daemon=True)
        prod_t.start()
        for t in post_ts:
            t.start()
        write_t.start()

        # [G] GPU forward stage runs on THIS (main) thread: the device was set in
        # main() here, and keeping a single GPU thread bounds VRAM to one forward(!)
        g_wait_in = g_fwd = g_wait_out = 0.0
        n_fwd = 0
        while True:
            s = pc()
            item = q_get(gpu_q)
            g_wait_in += pc() - s
            if item is _ABORT or item is _DONE:
                break
            ids_t, mask_t, chunk = item
            try:
                s = pc()
                with torch.no_grad():
                    logits = clf.model(input_ids=ids_t.to(device),
                                       attention_mask=mask_t.to(device)).logits
                # fp32 on host: postprocess runs a numpy softmax, which cannot take bf16.
                # .to("cpu") synchronises, so g_fwd is the GPU stage's true wall time.
                logits = logits.to(dtype=torch.float32, device="cpu")
                g_fwd += pc() - s
                n_fwd += 1
            except Exception as exc:
                fail("forward", exc)
                break
            s = pc()
            ok = q_put(post_q, (logits, chunk))
            g_wait_out += pc() - s
            if not ok:
                break
        add_times({"gpu_wait_in": g_wait_in, "gpu_fwd": g_fwd,
                   "gpu_wait_out": g_wait_out, "gpu_forwards": float(n_fwd)})

        # clean shutdown: one _DONE per postprocess worker, then one for the writer
        # on abort these are skipped; every thread exits via its abort-aware get
        if not abort.is_set():
            for _ in post_ts:
                q_put(post_q, _DONE)
        for t in post_ts:
            t.join()
        if not abort.is_set():
            q_put(write_q, _DONE)
        write_t.join()
        prod_t.join()

        if errors:
            stage, exc = errors[0]
            raise RuntimeError(f"pipeline stage {stage!r} failed on {in_path.name}: {exc!r}") from exc

        out.flush()

    os.replace(tmp_out, final_out)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if on_gpu else 0.0
    n_trunc = stats["n_trunc"]
    extra = f", {n_trunc} truncated@{safe_single}tok" if n_trunc else ""
    n_tokens = stats["n_tokens"]  # TEMP
    tok_s = n_tokens / dt if dt > 0 else 0.0  # TEMP: per-file throughput
    log(rank,
        f"done {in_path.name}: {stats['n_rows']} rows -> {stats['n_entities']} entities "
        f"in {dt:.1f}s (peak {peak:.1f}GB{extra}) [{n_tokens} tokens, {tok_s:,.0f} tok/s]")
    # GPU keys are this-thread wall time. prod_* is the single producer thread
    # post_* is SUMMED across N_POSTPROCESS_WORKERS
    log(rank, f"timers {in_path.name}: "
        + " ".join(f"{k}={timers.get(k, 0.0):.1f}"
                   for k in ("gpu_wait_in", "gpu_fwd", "gpu_wait_out",
                             "prod_read", "prod_tok", "prod_makejob", "prod_gpuq_put_wait",
                             "post_q_get_wait", "post_proc", "post_writeq_put_wait",
                             "write_q_get_wait", "write_io"))
        + f" gpu_forwards={int(timers.get('gpu_forwards', 0))}"
        + f" jobs={int(timers.get('prod_jobs', 0))}")
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

    # pin this rank to the CPU cores in the same NUMA domain as its GCD:
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
