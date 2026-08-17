import subprocess
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpw.ollama import BASE, ensure_models, generate
from tpw.power import PowerSampler, detect_mode, diagnose


def _shell(*cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def loaded_models() -> dict[str, str]:
    """Model name -> processor split, from Ollama's running-model endpoint."""
    try:
        running = requests.get(f"{BASE}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return {}
    out = {}
    for entry in running:
        total = entry.get("size", 0)
        on_gpu = entry.get("size_vram", 0)
        share = 100 * on_gpu / total if total else 0
        out[entry["name"]] = f"{share:.0f}% GPU ({on_gpu / 1e9:.1f}/{total / 1e9:.1f} GB)"
    return out


def main(config_path: str | None = None) -> None:
    cfg_path = Path(config_path) if config_path else ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gen = dict(cfg["generation"])

    print(f"GPU power: {diagnose()}")

    digests = ensure_models(cfg["models"])
    for tag, digest in digests.items():
        print(f"{tag:24s} {digest[:19]}")

    print("\nsmoke test")
    header = f"{'model':24s} {'reply':24s} {'tok':>4s}  {'energy':>10s}  placement"
    print(header)
    for model in cfg["models"]:
        with PowerSampler() as sampler:
            result = generate(model, "Reply with the single word: ok",
                              **{**gen, "num_predict": 128})
        trace = sampler.trace()
        energy = f"{trace.energy_j:.2f} J" if trace else "no reading"
        placement = loaded_models().get(model, "not reported")
        cut = " TRUNCATED" if result.get("done_reason") == "length" else ""
        reply = result["response"].strip().replace("\n", " ")[:24]
        print(f"{model:24s} {reply:24s} {result['eval_count']:>4d}  "
              f"{energy:>10s}  {placement}{cut}")

    mode, detail = detect_mode()
    lines = [
        f"power_mode\t{mode}",
        f"power_detail\t{detail}",
        f"ollama\t{_shell('ollama', '--version')}",
        f"gpu\t{_shell('nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader')}",
        f"num_ctx\t{gen.get('num_ctx')}",
        "",
        *(f"{tag}\t{digest}" for tag, digest in digests.items()),
    ]
    out = ROOT / "results" / "environment.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])