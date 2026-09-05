"""
Real-data access attempt, run and logged verbatim (Sec 3.6 of the paper).
Writes network_probe_log.txt with the actual stdout/stderr of each step.
"""
import subprocess
import sys
import urllib.request

log_lines = []


def log(line):
    print(line)
    log_lines.append(line)


log("=== Step 1: pip install datasets ===")
try:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "datasets"],
        capture_output=True, text=True, timeout=120,
    )
    log(f"returncode={r.returncode}")
    log(r.stdout[-1500:])
    log(r.stderr[-1500:])
except Exception as e:
    log(f"EXCEPTION: {type(e).__name__}: {e}")

log("\n=== Step 2: datasets.load_dataset against a HF-hosted sycophancy set ===")
try:
    import datasets
    ds = datasets.load_dataset("Anthropic/model-written-evals", split="train")
    log(f"SUCCESS: loaded {len(ds)} rows")
except Exception as e:
    log(f"EXCEPTION: {type(e).__name__}: {e}")

log("\n=== Step 3: direct HTTPS probes ===")
for host in ["https://pypi.org", "https://huggingface.co", "https://raw.githubusercontent.com"]:
    try:
        with urllib.request.urlopen(host, timeout=10) as resp:
            log(f"{host}: SUCCESS status={resp.status}")
    except Exception as e:
        log(f"{host}: EXCEPTION {type(e).__name__}: {e}")

with open("network_probe_log.txt", "w") as f:
    f.write("\n".join(log_lines))

log("\nLog written to network_probe_log.txt")
