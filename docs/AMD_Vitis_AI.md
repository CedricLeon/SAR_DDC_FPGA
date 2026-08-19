# AMD Vitis AI — tools & quirks catalog

> A running, plain-language catalog of the AMD Vitis AI tools, commands, and gotchas we hit in this
> project. Not exhaustive documentation — just "what does this tool give us, how do we call it, where to
> read more." Refine as we learn. Board = ZCU102, DPU = 3× DPUCZDX8G B4096 @ 300 MHz, Vitis AI 3.0.

---

## Tools at a glance

| Tool | What it gives us | Runs where |
| --- | --- | --- |
| **`xdputil`** | *Static* per-subgraph size (ops, memory) + isolated per-subgraph peak FPS; DPU core inventory | board |
| **`vaitrace`** | *Dynamic* per-**core** per-subgraph profile: HW time, efficiency %, DDR traffic split + a timeline | board |
| **Vitis Analyzer** | GUI that opens the `vaitrace --va` trace (per-core timeline) | host/desktop |
| **VART / XRT** | The runtime our C++ uses to run subgraphs on the DPU cores | board (library) |

---

## `xdputil` — static model info + micro-benchmark

Static, instant, no full run needed. We use three subcommands (see `scripts/fpga/benchmark/collect_roofline.py`):

- **`xdputil xmodel <model>.xmodel -l`** → JSON, one entry per DPU subgraph with `workload` (ops) and
  `reg info` = the memory footprint split: **REG_0** weights+bias, **REG_1** workspace/scratch, **REG_2**
  input buffer, **REG_3** output buffer. This is the *design-time* memory requirement (the "how big is
  this subgraph" question). Saved by us into `results/benchmark_hardware/_roofline/*_xmodel_info.json`.
- **`xdputil benchmark <model>.xmodel -i <index> 1`** → the isolated **peak FPS** of one subgraph, single
  thread, 60 s, zero host overhead — the pure-DPU ceiling for that subgraph.
- **`xdputil query`** → DPU arch/core inventory. **Quirk:** reports **4** cores but only **0–2** are real
  B4096; core 3 is an empty placeholder slot (fingerprint 0). vaitrace prints a harmless
  `Unsupported platform fingerprint: 0, cu_idx: 3` warning because of it.

## `vaitrace` — the dynamic DPU profiler

`/usr/bin/vaitrace` on the board. Wraps a program, runs it, and reports what the DPU (and VART) actually did.
**It attributes work to each individual DPU core** (`DPUCZDX8G_1/_2/_3`) — so it
answers "which core, how busy, how efficient, how much DDR," per subgraph, in hardware.

**Invocation** (recovered): `vaitrace [-t sec] [-o out.txt] [-c config.json] [MODE] <program> <args>`.
`-t` = max trace seconds (default 60); the program can finish sooner. One example we run:

```bash
vaitrace -t 160 --txt_summary -o /tmp/s.txt ./build_cpp/stream_pipeline \
  --xmodel active_model/*.xmodel --params active_model/entropy_params \
  --tile data/full_scene_i16.npy --out /tmp/o.ddc --windowed --p0 --fanout --threads 6 --neon --max-rows 3
```

**Output modes** (pick one):

| Flag | Output |
| --- | --- |
| `--txt_summary` / `--json_summary` | The per-core per-subgraph table (below), to stdout or `-o` file |
| `--va` | Vitis Analyzer trace: `xrt.run_summary` + `profile_summary.csv` + **`vart_trace.csv`** (per-event timeline) |
| `--fine_grained` | Detailed trace, capped at 10 s |
| `--xat` | Raw data (debug) |

**The `--txt_summary` DPU table** — one row per (**DPU core** × subgraph):

`DPU Id · Bat · SubGraph · WL · SW_RT · HW_RT · Effic · LdWB · LdFM · StFM · AvgBw` (all defined in the
glossary). It also prints a **CPU Functions** section (VART/XRT calls like `xir::XrtCu::run`) with run
counts + average time — the CPU-side dispatch view, though *not* our own pipeline stages unless added via
the config's custom-trace list.

**Per-layer mode:** finer than per-subgraph — either `--fine_grained` or the config JSON `runmode`
`"normal" → "debug"` runs the model subgraph-by-subgraph. **Caveat:** it needs the model *compiled* in
fine-grained mode (per-layer `mc_code`); our production (fused) xmodels warn `has no mc_code` and won't
profile per-layer without a recompile.

**Limitation to remember:** the table is aggregate over the run (per core, per subgraph); the *time
ordering* (when each core is busy) lives in `vart_trace.csv` / the Vitis Analyzer timeline, not the text
table. Bandwidth comes from the on-chip **APM** (AXI Performance Monitor) — you'll see `APM Stop
Collecting` in the log.

## Vitis Analyzer — the `--va` GUI

Host GUI from the full **Vitis** install (`/tools/Xilinx/Vitis/2024.1/bin/vitis_analyzer`; not in conda
or the Vitis-AI docker). Use it: `source /tools/Xilinx/Vitis/2024.1/settings64.sh`, then
`vitis_analyzer <dir>/xrt.run_summary`. It's a Qt GUI needing a display — over a headless remote (VSCode
Remote-SSH has no `$DISPLAY`) open it through an **X2Go** desktop session on the host (roundabout, but
works).

In practice the Unified-IDE view shows mainly the **Timeline**: one row per DPU core (`DPU_0/1/2`), each
DPU run a bar tagged with thread-id, workload, batch, and efficiency. **The bar length is the *software*
span (queue-wait + compute), not HW_RT.** The **DPU Summary** table and **DDR read/write-rate graphs**
UG1414 advertises did not appear in this view — they live in the `profile_summary.csv` / `summary.csv`
files we already have (and maybe the `--classic` analyzer). Those are plain CSVs, so we can also
parse/plot the timeline without the GUI.

## VART / XRT (runtime, for context)

**VART** (Vitis AI RunTime) is the C++ API our `stream_pipeline` uses to run subgraphs; **XRT** (Xilinx
RunTime) is the layer under it that owns the DPU cores (exposed as CUs `DPUCZDX8G_1/_2/_3`). VART assigns
runners to cores by a deterministic creation-order round-robin (the basis of our fan-out pinning). XRT
also has its own native profiler via an `xrt.ini` file (kernel/AXI traces) — noted, not yet used.

---

## Glossary (plain language)

### DPU hardware & model

| Term | Meaning · example |
| --- | --- |
| **DPU** | The neural-net accelerator block on the FPGA. Ours is **DPUCZDX8G**, the Zynq UltraScale+ family. |
| **B4096** | DPU size = 4096 multiply-adds per clock per core. 3 cores × 4096 × 0.3 GHz ≈ **1229 GOP/s** each. |
| **core** | One of the **3** parallel engines in our DPU. vaitrace names them `DPUCZDX8G_1/_2/_3`. |
| **CU (Compute Unit)** | XRT's word for a hardware kernel instance. Each DPU core shows up as one CU. |
| **subgraph** | A slice of the network the compiler maps to one DPU call — our `g_a`, `h_a`, `h_s`, `g_s`. |
| **xmodel** | The compiled model file the DPU executes. |
| **GOP / MAC** | GOP = billion operations. A MAC (multiply-accumulate) counts as **2** ops. |
| **fixpos** | The INT8 quantization scale (fixed-point position); `fixpos 4` means values are ÷16. |

### `xdputil xmodel -l` (static memory footprint)

| Term | Meaning · example |
| --- | --- |
| **workload_ops** | Compute size of the subgraph (MACs). ResSHyp `g_a` ≈ 40 GOP; `h_a` ≈ 0.28 GOP. |
| **REG_0 / const_bytes** | **Weights + bias** stored in DDR. ResSHyp `g_a` ≈ 3.7 MB. |
| **REG_1 / workspace_bytes** | Scratch memory for intermediate results. `g_a` ≈ 6.3 MB. |
| **REG_2 / input_bytes**, **REG_3 / output_bytes** | Input / output buffer sizes. |

### `vaitrace` summary columns (measured, per core × subgraph)

| Term | Meaning · example |
| --- | --- |
| **WL (Workload)** | GOP actually executed. |
| **HW_RT** | *Hardware* run time — the time the DPU core truly spent computing, from a **hardware counter**. The clean "pure compute" number. ResSHyp `g_a` ≈ 33.6 ms. |
| **SW_RT** | *Software* run time — wall time as seen in software, so it includes the overhead of handing the job to the DPU. Always ≥ HW_RT. |
| **Effic (Efficiency)** | Achieved GOP/s ÷ the DPU's peak GOP/s, as a %. "How full is the DPU while this runs." Big convs `g_a`/`g_s` ≈ 90–96%; tiny `h_a`/`h_s` ≈ 20–28% (too small to fill it). |
| **Perf** | Achieved speed in GOP/s (= WL ÷ HW_RT). |
| **LdWB (Load Weight+Bias)** | DDR **read** for weights, MB. Dominates the small `h_a`/`h_s` (they're weight-load-bound). |
| **LdFM (Load Feature Map)** | DDR **read** for the input activation, MB. |
| **StFM (Store Feature Map)** | DDR **write** for the output activation, MB. |
| **AvgBw** | Average DDR bandwidth = (LdWB+LdFM+StFM) ÷ HW_RT, MB/s. All our subgraphs sit ≪ the 17 GB/s ceiling. |
| **DPU Id / CU Full Name** | **Which core ran it** (`DPUCZDX8G_1/_2/_3`) — the per-core key that makes the table per-core, not just per-subgraph. |
| **APM (AXI Performance Monitor)** | The on-chip hardware counter vaitrace reads for the DDR-bandwidth numbers. |

### Runtime / timeline

| Term | Meaning · example |
| --- | --- |
| **VART** | Vitis AI RunTime — the C++ API our code calls to run a subgraph on the DPU. |
| **XRT** | Xilinx RunTime — the layer under VART that owns the CUs/cores. |
| **HAL** | Hardware Abstraction Layer — the lowest-level device calls. |
| **VART / HAL / DPU events** | Event categories vaitrace *can* record. Our default `--va` traces only capture **VART runner events** (one bar per DPU run, tagged with its core); HAL and raw hardware-DPU events aren't captured unless the trace config is expanded. |
| **vart_trace.csv** | The per-event timeline file `--va` produces (each DPU run with start/end + which core). |

---

## Resources

- UG1414 *Vitis AI User Guide* — profiling: [Profiling the Model](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Profiling-the-Model),
  [vaitrace Usage](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/vaitrace-Usage),
  [Text Summary](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Text-Summary),
  [Configuration](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Configuration).
- vaitrace examples (GitHub): [examples.md](https://github.com/Xilinx/Vitis-AI/blob/v3.0/examples/vai_profiler/examples.md) ·
  tutorial: [Vitis-AI-Tutorials #16](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/16-profiler_introduction/README.md).
- Same-DPU per-layer profiling write-up (Kria KV260, DPUCZDX8G): [partenit.io](https://partenit.io/measuring-what-actually-matters-per-layer-dpu-profiling-on-kria-kv260-with-mobilenet-and-resnet-50/).
- DPU internals / cores / ports: [Vitis AI system-integration](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-system-integration.html).
- In-repo: `scripts/fpga/benchmark/collect_roofline.py` (xdputil), `stream_roofline.py` (roofline from vaitrace numbers), `results/benchmark_stream/vaitrace/` (our captures).
