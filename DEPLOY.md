# Running the sweep

The full sweep is **30 jobs** (3 indices × {lstm, hybrid} × 5 variants), plus 9
fast ARIMA jobs. Measured cost: **3.45 s per trial**, 20 trials × 19 walks per
job → **~22 min per job single-threaded**, i.e. **~11 core-hours** total.

## One rule: one thread per job

These models are tiny (25–500 hidden units, batch 32, 4 input features), so the
GEMMs are far below the size where threading pays. Measured per optimizer step:

| model | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| 25 × 1 layer | 0.98 ms | 1.01 | 1.01 |
| 100 × 2 layers | **3.58 ms** | 5.01 | 5.70 |
| 250 × 2 layers | **9.27 ms** | 9.95 | 10.58 |
| 500 × 2 layers | 25.00 ms | **24.32** | 25.62 |

Four threads made the 100×2 model **59% slower**. `run_all.sh` pins every
worker to one thread and gets its parallelism from running many jobs at once.
A GPU does not help either, for the same reason.

## Local

```bash
scripts/run_all.sh              # concurrency defaults to cores-2
scripts/run_all.sh -n           # dry run: show the plan
scripts/run_all.sh -j 16        # pick concurrency explicitly
```

Finished jobs (those with a `metrics.csv`) are skipped, so re-running resumes.
`-f` forces a re-run. Failures are listed at the end and logged per job under
`logs/`. Anything after `--` is forwarded to `run_experiment.py`:

```bash
scripts/run_all.sh -m lstm -f -- --trials 2 --epochs 2 --max-walks 1   # smoke
```

On an 18-core machine the whole sweep is ~36 minutes.

## RunPod CPU pod

Compute-Optimized CPU pods go up to **32 vCPU / 64 GB** at **$0.96/hr**, and
pricing is linear at **$0.03 per vCPU-hour**:

| vCPU | RAM | $/hr |
|---|---|---|
| 2 | 4 GB | 0.06 |
| 4 | 8 GB | 0.12 |
| 8 | 16 GB | 0.24 |
| 16 | 32 GB | 0.48 |
| 32 | 64 GB | 0.96 |

Because the price is linear, the **total** bill is ~(core-hours x $0.03)
whatever size you pick — a bigger pod just spends it faster. So take the 32
vCPU. Expect roughly $0.5 for the sweep and ~$1 for the sweep plus a
returns-target comparison run. 64 GB is ample: each worker peaks under 1 GB.

Note the 3 GHz / 5 GHz selector. This workload is latency-bound and
single-threaded, so **clock speed maps almost directly to wall time**. If 5 GHz
costs less than 1.67x per vCPU, it is the better buy. Calibrate on the pod
before launching the sweep:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -c "
import time, warnings; warnings.filterwarnings('ignore')
import torch; torch.set_num_threads(1)
from src import data, wfo
t = time.perf_counter()
wfo.run_lstm_wfo(data.build_features('^GSPC'), n_trials=4, max_walks=1,
                 device='cpu', seed=0)
p = (time.perf_counter() - t) / 4
print(f'{p:.2f}s/trial (M5 Pro baseline 3.45) -> {p*380/60:.0f} min per job')
"
```

Multiply by 380 trials/job x 30 jobs to get the real total before committing.

1. **Create the pod.** Compute-Optimized, 32 vCPU. Choose a plain
   Ubuntu/Python image, *not* a PyTorch/CUDA template — the CUDA wheels are
   2–3 GB and useless here.

2. **Get the code and data across.** The repo is 15 MB including the cached
   CSVs. Either `git clone` (if you have pushed a remote) or:

   ```bash
   rsync -az --exclude .venv --exclude results --exclude logs \
         ./ root@<pod-host>:/workspace/lstm/
   ```

   `runpodctl send` / `runpodctl receive` also works if you would rather not
   set up SSH keys.

   **Ship `data/` — do not re-download.** `data/` is gitignored, but Yahoo
   revises history, so `download_data.py` on the pod can return a different
   series and make the results incomparable with anything produced locally.

3. **Install.** CPU-only torch, or pip drags in the whole CUDA stack:

   ```bash
   cd /workspace/lstm
   python3 -m venv .venv && . .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   pytest tests/ -q                 # expect 56 passed
   ```

4. **Run**, detached so an SSH drop does not kill it:

   ```bash
   nohup scripts/run_all.sh -j 30 > logs/sweep.log 2>&1 &   # leave 2 cores
   tail -f logs/sweep.log
   ```

5. **Collect.**

   ```bash
   rsync -az root@<pod-host>:/workspace/lstm/results/ ./results/
   ```

   Then locally: `scripts/run_ensemble.py` and `scripts/make_report.py`.

Put the work under `/workspace` (the persistent volume). Anything written
elsewhere in the container is lost when the pod stops.

## Sharding across several machines

```bash
scripts/run_all.sh -s 0,3        # machine A: jobs 0, 3, 6, ...
scripts/run_all.sh -s 1,3        # machine B
scripts/run_all.sh -s 2,3        # machine C
```

Merge by rsyncing each machine's `results/` into one tree; the per-job
directories do not collide.
