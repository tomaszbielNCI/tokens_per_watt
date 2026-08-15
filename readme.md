# tokens_per_watt

Energy measurement of local LLM inference under a single-user,
batch-size-1 regime. Compares four efficiency metrics on the same runs
to test whether they yield consistent model rankings.

## Requirements

- Python 3.12 (Anaconda `base` environment)
- Ollama (serves the models; must be running on `localhost:11434`)
- NVIDIA GPU with `nvidia-smi` reporting `power.draw` — optional;
  the power sampler degrades to a no-op on machines without one,
  so the pipeline can be developed on a CPU-only laptop.

## Setup (Windows)

Install Ollama:

    winget install Ollama.Ollama

Restart the terminal — or the whole IDE if using the PyCharm terminal,
since PyCharm inherits environment variables at startup and will not
see the updated `PATH` until it is relaunched.

Pull a small model for smoke testing:

    ollama pull qwen2.5:0.5b

Verify the server responds:

    curl.exe http://localhost:11434/api/tags

In PowerShell, use `curl.exe` rather than `curl`: the latter is an
alias for `Invoke-WebRequest` and takes different arguments.

The Ollama tray application is the server. If the tray icon is absent,
no process is listening on port 11434 and every request will fail with
`WinError 10061`.

## Setup (Linux / Kaggle)

    curl -fsSL https://ollama.com/install.sh | sh

See `notebooks/` for starting the server from a notebook cell.

## Measurement notes

Set `OLLAMA_KEEP_ALIVE=-1` before any measurement run. Ollama unloads
models after five minutes of inactivity by default, which would charge
the reload cost to whichever run happens to follow an idle gap.

Pin `num_ctx` explicitly and identically across models — the default is
derived from model metadata and varies, changing KV cache allocation
and therefore the power profile.

Pin quantisation explicitly in the model tag (e.g.
`qwen2.5:7b-instruct-q4_K_M`) and record the digest from
`ollama list --digest`, so that a tag reassigned upstream does not
silently change what was measured.

## Layout

    tpw/contracts.py   dataclasses shared by every other module
    tpw/power.py       GPU power sampling and energy integration
    tpw/ollama.py      generation client
    tpw/tasks.py       task loading and difficulty stratification
    tpw/grading.py     answer parsing and scoring
    tpw/runner.py      grid loop with resume-from-CSV
    tpw/metrics.py     aggregation across metrics
    scripts/           entry points
    notebooks/         Kaggle and Colab runners

No module imports another except `contracts`, so any single file can be
pasted into a notebook cell and run on its own.