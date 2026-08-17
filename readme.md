# tokens_per_watt

Energy measurement of local LLM inference under a single-user,
batch-size-1 regime. The same runs are scored under four efficiency
metrics — joules per token, joules per correct answer, tokens per
correct answer, and a dollar figure computed the way vendors report it —
to test whether those metrics rank models consistently.

## Requirements

- Python 3.12 (Anaconda `base` environment)
- Ollama serving on `localhost:11434`
- `nvidia-ml-py`, `datasets`, `pandas`, `pyyaml`
- An NVIDIA GPU exposing power telemetry through NVML — optional; the
  sampler degrades to a no-op, so the pipeline can be developed on a
  machine with no GPU.

## Power telemetry

NVML offers three routes to power, and which one works varies by card.
`tpw/power.py` probes them in order and reports the mode it settled on:

1. `NVML_TOTAL_POWER_SAMPLES` — a buffer the driver fills at roughly
   50 Hz. Energy is the trapezoid integral over the drained samples.
2. `nvmlDeviceGetTotalEnergyConsumption` — a cumulative millijoule
   counter; energy is the difference between two reads.
3. `nvmlDeviceGetPowerUsage` — polled instantaneous power, integrated
   the same way as route 1.

On the GeForce RTX 4060 used here, routes 2 and 3 both raise
`NVML_ERROR_NOT_SUPPORTED` and `nvidia-smi --query-gpu=power.draw`
returns `N/A`, while route 1 works. Any tooling built on the
`power.draw` field alone would classify this card as unmeasurable.
Run `scripts/preflight.py` to see which route is active before
trusting any energy figure.

## Setup (Windows)

    winget install Ollama.Ollama

Restart the terminal afterwards — or the whole IDE if using the PyCharm
terminal, since PyCharm inherits environment variables at startup and
will not see the updated `PATH` until it is relaunched.

The Ollama tray application is the server. Without the tray icon,
nothing listens on port 11434 and every request fails with
`WinError 10061`. Verify with:

    curl.exe http://localhost:11434/api/tags

Use `curl.exe`, not `curl`: the latter is a PowerShell alias for
`Invoke-WebRequest` and takes different arguments.

Measurement runs should be launched from a plain shell rather than from
an IDE terminal. PyCharm's embedded browser holds GPU memory and raises
the idle power floor, which is subtracted from every measurement.

## Setup (Linux / Kaggle)

    curl -fsSL https://ollama.com/install.sh | sh

See `notebooks/` for starting the server from a notebook cell.

## Measurement notes

**Model residency.** `keep_alive: -1` disables the idle unload timer but
does not prevent eviction when VRAM fills. On an 8 GB card the model set
cannot be co-resident, so `build_grid` groups cells by model and
randomises only within each group, and `run_grid` issues an unmeasured
warm-up call on every model change. Without this, the VRAM load cost —
125 J for a 0.5B model, 927 J for a 7B one — lands on whichever task
happens to follow a switch.

**Context window.** `num_ctx` is pinned identically across models.
Defaults are read from model metadata and differ widely (262144 for
gemma4:12b against 32768 for mistral:latest), changing KV cache
allocation and with it the power profile.

**Model identity.** Floating tags such as `mistral:latest` may be
reassigned upstream. `scripts/preflight.py` records the digest reported
by `/api/tags` for every model into `results/environment.txt` alongside
the driver, Ollama version, and pinned `num_ctx`.

**Idle floor.** A request costs about 2.2 s and 130 J even when the
model emits three tokens, so a large fraction of any short run is the
card idling. Measure the baseline with `scripts/idle_baseline.py` in the
same desktop state the grid will run in, and report both the raw and
idle-subtracted figures.

**Placement.** `scripts/preflight.py` prints the CPU/GPU split from
`/api/ps`. Anything short of 100% GPU means part of the work ran on the
CPU, which NVML does not see, and that run is not comparable.

## Running

    python scripts/preflight.py        # pull models, probe telemetry, record environment
    python scripts/run_grid.py config_test.yaml
    python scripts/audit_parsing.py    # inspect parser failures before the full grid
    python scripts/idle_baseline.py 300
    python scripts/run_grid.py         # full grid

`run_grid` appends each row as it completes and skips cells already
present in the output CSV, so an interrupted run resumes and a widened
grid computes only the new cells.

## Answer parsing

Prompts request a trailing `ANSWER: <number>` marker. Parsing prefers
that marker and falls back to the last number in the visible output, so
that failure to honour the format is not scored as a wrong answer.
Compliance is recorded separately in `format_ok`, since it varies
sharply with model size and is a result in its own right.
Reasoning-model `<think>` blocks are stripped before parsing and their
length recorded, since they are the verbalisation overhead under test.

## Layout

    tpw/contracts.py       dataclasses shared by every other module
    tpw/power.py           NVML telemetry probing and energy integration
    tpw/ollama.py          generation client, model pulls, digests
    tpw/tasks.py           GSM8K loading, stratified by reference-solution steps
    tpw/grading.py         prompts, answer parsing, scoring
    tpw/runner.py          grid construction and the resumable run loop
    tpw/metrics.py         aggregation across the four metrics
    scripts/               entry points
    notebooks/             Kaggle and Colab runners
    config.yaml            model set, grid shape, generation parameters

No module imports another except `contracts`, so any single file can be
pasted into a notebook cell and run on its own.