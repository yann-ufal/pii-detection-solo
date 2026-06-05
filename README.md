## PII detection on LUMI

Massively parallel inference with the `openai/privacy-filter` PII-detection model over a large tree of `*.jsonl.zst(d)` files, designed to run across multiple full LUMI-G nodes inside a Singularity container, launched with `torchrun`.


### 1. What is this?

* Reads every `*.jsonl.zst(d)` file/symlink found anywhere under an input root folder.
* For each line (a JSON object), it looks at the `text` field and runs the model to find PII entities.
* A line is skipped only if its `propella-4b.pii_presence` value is `"no_pii"`.
  If the `propella-4b` key is missing, the line is processed.
* For every PII entity found, it writes one output line with the keys, in order:

  | key | type |  
  | --- | --- |  
  | `id`        | str |  
  | `name`      | str |  
  | `value`     | str |  
  | `start_pos` | int |  
  | `end_pos`   | int |  

* Output files mirror the input tree exactly: same subfolders, same filenames,
  written under the output root folder.

The model's labels used for `name` identifying `value` entity types are: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`.


### 2. Files in this project

| File | Purpose |
|---|---|
| `pii_infer.py`       | The inference program. One copy runs per GCD; each handles a balanced shard of files |
| `venv_and_squash.sh` | Run once on a login node to upgrade `transformers`, pre-download the model, and pack everything into `user-software.sqsh` |
| `run_it.sh`          | The Slurm batch script. Submit it with `sbatch run_it.sh` |
| `pyproject.toml`     | Dependency list for optional local testing only |


### 3. Nice things

Resume: each output file is written to a `.tmp` file and atomically renamed only on success. If a job hits the time limit and you resubmit, files that already exist are skipped, so you resume roughly where you left off. You loose the files being processed as .tmp when the job expired and the .tmp remain on disk.


### 4. One-time setup on LUMI  

#### 4.1 Log in and get the code there

```bash
ssh <your-username>@lumi.csc.fi  
cd /scratch/project_XXXXXXXXX/users  
scp these three files to a folder somewhere nice at this location  
cd there  
```

#### 4.2 Edit the hardcoded paths and account

Open the files and set the values for your project:

* `pii_infer.py` — set `INPUT_ROOT` and `OUTPUT_ROOT`
* `run_it.sh` — set `#SBATCH --account=project_XXXXXXXXX`, `--nodes` and `--time`

> [!IMPORTANT]
> `--nnodes=1` is per-node-launcher and scaling is done purely via `#SBATCH --nodes`. Do NOT change.


#### 4.3 Build the SquashFS environment (login node, has internet)

Optionally authenticate first on hugging face (recommended on LUMI, where login nodes share one outbound IP and thus share the anonymous per-IP rate limit):

```bash
export HF_TOKEN=hf_xxx
```

```bash
bash venv_and_squash.sh
```

The token is used only here at build time; the compute-node job is fully offline.
When it finishes you will have `user-software.sqsh` in the project folder.
You do not need to run this again unless you want to update packages.

### 5. Run the job

```bash
sbatch run_it.sh
```

Check on it:

```bash
squeue --me                    # is it queued / running?
tail -f logs/pii-<jobid>.out   # live progress (per-rank log lines)
```

Each rank prints lines like:

```
[rank 0] assigned 27 files
[rank 0] done part-000123.jsonl.zst: 48211 rows -> 1502 entities in 73.4s
[rank 0] progress 10/27 files
```

When the job ends, your results are under `OUTPUT_ROOT`, in the same folder layout and with the same filenames as the input.


### 6. Knobs

In `pii_infer.py`

* `VRAM_CEILING_GB` (default `50.0`) The GCD has 64 GB
* `VRAM_TARGET_FRACTION` (default `0.95`) — plan batches/lengths against this fraction of the ceiling
* `DTYPE` (default `torch.bfloat16`) — model compute dtype
* `BUFFER_DOCS` (default `4096`) — how many documents are buffered before length-sorting and batching. Larger = less padding waste, more host RAM
* `MAX_BATCH_SEQS` (default `512`) — hard cap on sequences per forward, on top of the token budget
* `AGGREGATION` — how sub-word tokens are merged into entity spans. `"first"` by default

The startup VRAM calibration sizes batches and the `safe_single` length cap automatically from measured memory.
There is no `MAX_LENGTH`/`STRIDE` — this model is fed whole documents. If `WARNING: peak ... exceeded the ... ceiling` shows in the logs, lower `VRAM_TARGET_FRACTION`


## 7. Troubleshooting

* `squeue` shows nothing / job ended instantly — check `logs/pii-<jobid>.err`. The most common causes are a wrong `--account` or paths that don't exist.
* `OSError: ... openai/privacy-filter` / "Can't load" — the model wasn't baked into the `.sqsh`. Re-run `venv_and_squash.sh` on a login node (it needs internet) and confirm it printed the transformers version and "model cached".
* CUDA/HIP out of memory — lower `VRAM_TARGET_FRACTION`
* Job hit the time limit — just `sbatch run_it.sh` again. finished files are skipped and it resumes. Voila.

