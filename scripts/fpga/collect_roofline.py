#!/usr/bin/env python3
"""collect_roofline.py — measure peak DPU FPS per subgraph via xdputil benchmark.

Run ON THE BOARD while the target model is active in active_model/.

Usage:
    python3 collect_roofline.py <model_name>

Output:
    /home/root/SAR_DDC/bench_results/<model_name>_xdputil_peaks.json

Each DPU subgraph is benchmarked with 1 thread for 60 s (xdputil fixed window).
The subgraph indices are discovered dynamically from `xdputil xmodel -l` — they
are NOT sequential (USER/CPU subgraphs are interspersed). Functional names
(g_a, h_a, h_s, g_s) are extracted from the subgraph name field.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

BOARD_ROOT = Path("/home/root/SAR_DDC")
FUNC_NAMES = ["g_a", "h_a", "h_s", "g_s"]


def detect_func_name(sg_name: str):
    """Detect functional name (g_a/h_a/h_s/g_s) from subgraph name."""
    for fn in FUNC_NAMES:
        if f"_{fn}_" in sg_name:
            return fn
    return None


def list_dpu_subgraphs(xmodel: Path):
    """Return list of dicts with index, func, name, and static metadata (ops, bytes)."""
    raw = subprocess.check_output(
        ["xdputil", "xmodel", str(xmodel), "-l"],
        stderr=subprocess.DEVNULL,
    )
    data = json.loads(raw)
    dpu_sgs = []
    for sg in data["subgraphs"]:
        if sg.get("device") != "DPU":
            continue
        fn = detect_func_name(sg["name"])
        if fn is None:
            raise RuntimeError(
                f"Cannot detect functional name (g_a/h_a/h_s/g_s) for DPU "
                f"subgraph: {sg['name']!r}. Update FUNC_NAMES or detect_func_name()."
            )
        dpu_sgs.append({"index": sg["index"], "func": fn, "name": sg["name"]})
    if not dpu_sgs:
        raise RuntimeError(f"No DPU subgraphs found in {xmodel}")
    return dpu_sgs


def benchmark_subgraph(xmodel: Path, index: int):
    """Run xdputil benchmark with 1 thread and return peak FPS."""
    result = subprocess.run(
        ["xdputil", "benchmark", str(xmodel), "-i", str(index), "1"],
        capture_output=True,
        text=True,
    )
    m = re.search(r"FPS=\s*([\d.]+)", result.stderr)
    if not m:
        raise RuntimeError(
            f"Could not parse FPS from xdputil output for subgraph index {index}.\n"
            f"stderr:\n{result.stderr}"
        )
    return float(m.group(1))


def main():
    """Entry point."""
    if len(sys.argv) != 2:
        sys.exit("Usage: collect_roofline.py <model_name>")
    model_name = sys.argv[1]

    active = BOARD_ROOT / "active_model"
    xmodels = list(active.glob("*.xmodel"))
    if len(xmodels) != 1:
        raise RuntimeError(f"Expected exactly one .xmodel in {active}, found: {xmodels}")
    xmodel = xmodels[0]

    outfile = BOARD_ROOT / "bench_results" / f"{model_name}_xdputil_peaks.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"model:  {model_name}")
    print(f"xmodel: {xmodel.name}")

    dpu_sgs = list_dpu_subgraphs(xmodel)
    print(f"DPU subgraphs: {[(s['func'], s['index']) for s in dpu_sgs]}")
    print()

    peaks = {}
    for sg in dpu_sgs:
        print(f"--- {sg['func']} (xmodel subgraph index {sg['index']}) --- ~60 s")
        fps = benchmark_subgraph(xmodel, sg["index"])
        peaks[sg["func"]] = fps
        print(f"    {fps:.3f} fps\n")

    out = {"model_name": model_name, "xmodel": str(xmodel), **peaks}
    outfile.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Written: {outfile}")


if __name__ == "__main__":
    main()
