# Performance Benchmark Implementation

> Documentation of the FPGA performance benchmarking framework for the SAR DDC model.
>
> Tracks hardware exploration, measurement methodology, and all data collected from
> the Xilinx ZCU102 evaluation board and AMD documentation.

---

## 1. Hardware Platform — Xilinx ZCU102

### 1.1 Board Overview

| Component | Specification |
|---|---|
| **SoC** | Zynq UltraScale+ XCZU9EG-2FFVB1156 |
| **CPU** | 4× ARM Cortex-A53 (ARMv8-A, ~200 BogoMIPS per core) |
| **DPU IP** | DPUCZDX8G v4.1 — ISA1 B4096 |
| **DPU Cores** | 3 (though benchmark uses 1 thread by default) |
| **DPU Frequency** | 300 MHz |
| **DPU Fingerprint** | `0x101000056010407` |
| **Vitis-AI Version** | 3.0 |
| **Batch Size** | 1 (DPU arch B4096) |
| **OS** | PetaLinux (aarch64) |
| **Python** | 3.8 (Vitis-AI constraint) |

### 1.2 DPU Architecture Notes

The B4096 architecture provides 4096 operations per clock cycle. At 300 MHz, the
theoretical peak throughput is:

$$
\text{Peak}_\text{INT8} = 4096 \times 300 \times 10^6 = 1.2288 \;\text{TOPS (INT8)}
$$

For convolutions, the DPU performs multiply-accumulate operations, so each "OP"
represents one MAC on INT8 data.

---

## 2. Model Architecture Analysis

### 2.1 ResidualScaleHyperpriorPatched

The model is a hyper-autoencoder based on the Scale Hyperprior (Ballé et al., 2018)
with residual blocks and ReLU activations (GDN replaced for DPU compatibility).

| Parameter | Value |
|---|---|
| **N** (main channels) | 128 |
| **M** (hyper channels) | 256 |
| **Input** | 256 × 256 × 1 (per-channel, real/imag processed separately) |
| **Activation** | ReLU (GDN not supported by Vitis-AI DPU) |
| **Total Parameters** | ~15M (including entropy bottleneck) |

### 2.2 Subgraph Breakdown

The model is split into 4 DPU subgraphs + CPU-based entropy coding:

| Subgraph | Role | Input Shape (NHWC) | Output Shape (NHWC) | Params (CONST bytes) | Workload (OPs) |
|---|---|---|---|---|---|
| **g_a** | Encoder | 1×256×256×1 | 1×16×16×128 | 3,731,456 | 39,754,825,728 |
| **h_a** | Hyper-encoder | 1×16×16×256 | 1×2×2×256 | 4,927,488 | 275,272,704 |
| **h_s** | Hyper-decoder | 1×2×2×256 | 1×16×16×256 | 3,158,016 | 176,246,784 |
| **g_s** | Decoder | 1×16×16×128 | 1×256×256×1 | 3,289,088 | 38,131,662,848 |
| **Total** | | | | **15,106,048** | **78,338,007,104** |

### 2.3 Why h_a and h_s Have Few OPs Despite Many Parameters

This is a key observation: **h_a has 4.9M parameters but only 275M OPs**, while
**g_a has 3.7M parameters but 39.75 GOPS**. The reason is the spatial dimension:

**OPs scale as:** $\text{OPs} \propto \text{kernel\_params} \times H_\text{out} \times W_\text{out}$

For a Conv2d with kernel $k \times k$, $C_\text{in}$ input channels, $C_\text{out}$
output channels, and output spatial size $H_\text{out} \times W_\text{out}$:

$$
\text{MACs} = k^2 \times C_\text{in} \times C_\text{out} \times H_\text{out} \times W_\text{out}
$$

| Subgraph | Typical Spatial Dims | Effect on OPs |
|---|---|---|
| g_a | 256×256 → 128×128 → 64×64 → 32×32 → 16×16 | Large feature maps → **high OPs** |
| h_a | 16×16 → 8×8 → 4×4 → 2×2 | Tiny feature maps → **low OPs** |
| h_s | 2×2 → 4×4 → 8×8 → 16×16 | Tiny feature maps → **low OPs** |
| g_s | 16×16 → 32×32 → 64×64 → 128×128 → 256×256 | Large feature maps → **high OPs** |

**Consequence for DPU**: h_a and h_s are extremely fast on DPU (sub-millisecond)
because the actual computation is minimal. Their large parameter count comes from
having many wide 5×5 kernels with 256 channels, but those kernels operate on just
2×2 or 4×4 spatial grids.

### 2.4 Pipeline Execution Flow

For one 256×256 tile, the full compress+decompress pipeline executes:

```
[DPU] g_a(real)       → y_real   [1,16,16,128]
[DPU] g_a(imag)       → y_imag   [1,16,16,128]
[CPU] concat + abs    → y        [1,16,16,256], y_abs [1,16,16,256]
[DPU] h_a(y_abs)      → z        [1,2,2,256]
[CPU] EB.compress(z)  → z_strings
[CPU] EB.decompress   → z_hat    [1,2,2,256]
[DPU] h_s(z_hat)      → scales   [1,16,16,256]
[CPU] GC.compress(y)  → y_strings
[CPU] GC.decompress   → y_hat    [1,16,16,256]
[CPU] split y_hat     → y_hat_real [1,16,16,128], y_hat_imag [1,16,16,128]
[DPU] g_s(y_hat_real) → recon_real [1,256,256,1]
[DPU] g_s(y_hat_imag) → recon_imag [1,256,256,1]
```

**Total DPU calls per tile: 6** (g_a×2, h_a×1, h_s×1, g_s×2).

---

## 3. xdputil — DPU Utility Tool

`xdputil` is a command-line utility that ships with the Vitis-AI runtime on the
board. It interfaces directly with the DPU hardware and compiled `.xmodel` files.

| Subcommand | What It Does |
|---|---|
| `xdputil query` | Reads DPU hardware registers: number of cores, architecture (B4096), clock frequency (300 MHz), fingerprint, Vitis-AI version. |
| `xdputil xmodel <model> -l` | **Static analysis** of a compiled `.xmodel`. Lists all subgraphs with their workload (OPs), memory regions (weights/workspace/IO sizes), tensor shapes, and fixed-point positions. No inference runs. |
| `xdputil benchmark <model> -i <idx>` | **Synthetic throughput benchmark** on one subgraph. Feeds random data to the DPU as fast as possible for ~15 s and reports peak FPS. Measures pure DPU throughput with zero CPU overhead. |
| `xdputil run <model>` | Single inference (debugging). |

### 3.0 Commands Used to Obtain Section 2 Data

All parameter counts, workload OPs, and memory sizes in Section 2.2 come from:

```bash
# On the ZCU102 board
xdputil xmodel /home/root/SAR_DDC/active_model/compiled_model/*.xmodel -l
```

This outputs JSON. Per DPU subgraph, the relevant fields are:
- `"workload"` → total OPs (values in the "Workload" column)
- `"reg info"` → memory regions: `REG_0` (CONST = INT8 weights), `REG_1`
  (WORKSPACE), `REG_2` (INPUT), `REG_3` (OUTPUT). The `"size"` field in
  `REG_0` gives parameter bytes (1 byte per INT8 weight).

The benchmark script parses this automatically via `--collect-hw-meta`
(see `get_xmodel_metadata()` in `benchmark_fpga.py`).

The subgraph DPU indices (needed for `xdputil benchmark -i <idx>`) also come
from this JSON output.

### 3.1 xdputil Benchmark Results

Single-thread, 15-second runs using `xdputil benchmark <xmodel> -i <index>`:

| Subgraph | DPU Index | Workload (OPs) | Peak FPS | Implied Latency (ms) | Computational Throughput |
|---|---|---|---|---|---|
| g_a | 10 | 39,754,825,728 | 27.8 | 35.97 | 1.105 TOPS |
| h_a | 6 | 275,272,704 | 1,225 | 0.82 | 0.337 GOPS |
| h_s | 4 | 176,246,784 | 1,375 | 0.73 | 0.242 GOPS |
| g_s | 8 | 38,131,662,848 | 28.5 | 35.09 | 1.087 TOPS |

### 3.2 DPU Utilization Analysis (Per-Core)

The benchmark uses **1 thread → 1 DPU core** (out of 3 available). The
theoretical peak of 1.2288 TOPS in Section 1.2 is the peak for **one** B4096
core. Utilization is therefore computed **per-core**:

$$
\eta_\text{DPU} = \frac{\text{Measured Throughput (single core)}}{\text{Peak Throughput (single core)}} = \frac{1.105}{1.229} \approx 89.9\%
$$

This is excellent utilization, expected for large convolution-dominated models
operating on sufficiently large feature maps.

For h_a/h_s, the utilization appears very low because the spatial dimensions are
too small to fully occupy the B4096 processing elements. This is expected and not
a concern since these subgraphs contribute negligible latency (~1.5 ms combined
vs ~71 ms for g_a + g_s).

**Multi-core note**: Using all 3 DPU cores in parallel (3 threads) could yield
up to ~3× system throughput (theoretical max: $3 \times 1.23 = 3.69$ TOPS), but
the per-core utilization would remain ~90%. Multi-core benchmarking is a future
work item (Section 9).

### 3.3 xmodel Memory Summary

From `xdputil xmodel <xmodel> -l`:

| Subgraph | CONST (weights) | WORKSPACE | INPUT | OUTPUT |
|---|---|---|---|---|
| g_a | 3,731,456 B (3.56 MB) | 6,291,456 B (6.00 MB) | 66,320 B | 32,768 B |
| h_a | 4,927,488 B (4.70 MB) | 20,480 B | 65,536 B | 1,024 B |
| h_s | 3,158,016 B (3.01 MB) | 20,480 B | 1,024 B | 65,536 B |
| g_s | 3,289,088 B (3.14 MB) | 8,257,536 B (7.87 MB) | 32,768 B | 65,536 B |

**Total weight memory**: ~14.4 MB (INT8 quantized).

---

## 4. Power Measurement — INA226 Sensors

### 4.1 What Are INA226 Sensors?

The **TI INA226** is a small integrated circuit (chip) that acts as a **power
monitor**. It is soldered onto the ZCU102 printed circuit board (PCB) in series
with a tiny current-sensing resistor on a power rail. It measures:

- **Voltage** across the rail (bus voltage, in mV).
- **Current** flowing through the rail (derived from the voltage drop across
  the sense resistor, in mA).
- **Power** (voltage × current, reported in micro-watts).

Think of it as a tiny ammeter + voltmeter permanently wired into one of the
board's power lines. The chip digitises these measurements with a 16-bit ADC
and makes them available over the I2C bus, which Linux exposes as files in
`/sys/class/hwmon/`. Reading the file gives the instantaneous measurement:

```
cat /sys/class/hwmon/hwmon10/power1_input   # → 5968000 (µW = 5.968 W)
```

The ZCU102 board has **18 INA226 chips**, each monitoring a different power rail.

### 4.2 What Are Power Rails?

A "power rail" is a copper trace on the PCB that carries a specific supply
voltage to a group of circuits. The Zynq UltraScale+ SoC has dozens of power
pins that require different voltages (0.85 V, 1.2 V, 1.8 V, 3.3 V, etc.).
Each group of pins is fed by a dedicated **voltage regulator** (a chip that
converts the 12 V input into the required voltage). The copper trace from the
regulator's output to the SoC pins is the "power rail" and is given a net name
in the schematic (e.g., `VCCINT`, `VCCBRAM`).

**Why rails matter for us:**

| Rail | Nominal Voltage | What It Powers | Why We Care |
|---|---|---|---|
| **VCCINT** | 0.85 V | PL (programmable logic) core — **this is the DPU fabric** | Main indicator of DPU computation power |
| **VCCBRAM** | 0.85 V | Block RAM inside the PL | DPU uses BRAM for weights & activations |
| **VCCAUX** | 1.80 V | PL auxiliary / clocking circuits | Minor, mostly static |
| **VCCPSINTFP** | 0.85 V | PS full-power domain — **ARM Cortex-A53 cores** | CPU-side entropy coding power |
| **VCCPSINTLP** | 0.85 V | PS low-power domain (RPU, always-on logic) | Background, not our workload |
| **VCCO_PSDDR_504** | 1.20 V | DDR4 memory I/O interface | Data transfer overhead |

The other rails (PLL, MGT, operational supplies, FMC) are largely static and
not driven by our inference workload.

**Source for rail ↔ function mapping**: The "What It Powers" descriptions are
not guesses — they come from the official Xilinx/AMD documentation:
- **DS925** (Zynq UltraScale+ Data Sheet): "Recommended Operating Conditions"
  table, which defines each rail with a short description
  (e.g., VCCINT = "Internal core supply", VCC_PSINTFP = "PS full-power domain
  internal supply").
- **UG583** (PCB Design Guide): Groups rails into power domains — LPD
  (R5 RPU), FPD (A53 APU), PLPD (PL fabric), BPD (battery) — and documents
  which rails can be consolidated.
- **UG1085** (Technical Reference Manual): Defines what hardware lives in each
  power domain (FPD = A53 + GPU + SATA + PCIe; LPD = R5 + peripherals).

### 4.3 What Is a "Reference Designator" (Ref. Des.)?

Every component soldered onto a PCB is assigned a unique **reference designator**
— a short label like **U47**, **U79**, **R123**, **C45**. The letter indicates
the component type (U = IC/chip, R = resistor, C = capacitor) and the number is
just a sequential ID. It is printed on the PCB silkscreen next to the component
and used in the schematic to identify it.

On the ZCU102:
- **U47** is a MAX15301 voltage regulator that *generates* the VCCINT rail.
- **U79** is an INA226 power monitor that *measures* the VCCINT rail.

These are two different chips with different reference designators, but they are
both associated with the same power rail. This distinction is important because
the Linux sysfs sensor names use the **INA226 chip's** reference designator
(e.g., `ina226_u79`), not the regulator's.

### 4.4 What Is PMBus?

**PMBus** (Power Management Bus) is a standardised communication protocol built
on top of I2C (a simple 2-wire serial bus). It lets a host processor (the ARM
A53 in our case) talk to power management chips — both the Maxim voltage
regulators and the TI INA226 monitors — over the same physical wires.

On the ZCU102, all PMBus devices hang off the **I2C0 bus** through a
**PCA9544A 4-channel I2C mux** (chip U60). The mux has separate channels:
- **Channel 0 (MAXIM_PMBUS)**: Maxim voltage regulators (for programming
  voltage setpoints, enable/disable, reading regulator-reported telemetry).
- **Channel 1 (PS_PMBUS)**: INA226 monitors for PS-side rails.
- **Channel 2 (PL_PMBUS)**: INA226 monitors for PL-side rails.

Each device on a channel has a unique **I2C address** (e.g., 0x40, 0x41, ...),
which is a 7-bit number that identifies it on the bus. Two devices can have the
same address as long as they are on *different* mux channels — which is exactly
the case here: VCCINT's INA226 (U79) is at PL:0x40 and VCCPSINTFP's INA226
(U76) is at PS:0x40.

### 4.5 How We Map Sysfs Sensors to Power Rails

Linux exposes each INA226 as `/sys/class/hwmon/hwmonN/` with a `name` file
containing the sensor identifier (e.g., `ina226_u79`). To know which rail that
monitors, we need to cross-reference **two tables** from UG1182 (v1.7):

1. **Table 3-22** "I2C0 U60 Mux Target Bus Connections" — lists each INA226
   chip by its **PCB reference designator** (U79, U76, …) and **I2C address**
   (0x40, 0x41, …) on its bus channel (PL_PMBUS or PS_PMBUS).
2. **Table 3-56** "ZCU102 Power Rails with INA226 Power Monitors" — lists each
   monitored power rail, its **regulator** reference designator, and the
   **INA226 I2C address** on its bus channel.

Matching on `(bus channel, I2C address)` gives the definitive mapping:

```
sysfs "ina226_u79" → (Table 3-22) U79 on PL_PMBUS @ 0x40
                   → (Table 3-56) PL:0x40 = VCCINT
                   → U79 monitors VCCINT ✓
```

#### PL_PMBUS (I2C mux channel 2) — PL / DPU side

| INA226 Chip | I2C Addr | Rail | Regulator | Rail Voltage | Description |
|---|---|---|---|---|---|
| **U79** | 0x40 | **VCCINT** | U47 (MAX15301) | 0.85 V | **PL core — DPU fabric** |
| U81 | 0x41 | VCCBRAM | U7 (MAX15303) | 0.85 V | PL Block RAM |
| U80 | 0x42 | VCCAUX | U6 (MAX15303) | 1.80 V | PL auxiliary |
| U84 | 0x43 | VCC1V2 | U10 (MAX15303) | 1.20 V | PL 1.2 V supply |
| U16 | 0x44 | VCC3V3 | U9 (MAX15303) | 3.30 V | PL 3.3 V I/O |
| U65 | 0x45 | VADJ_FMC | U63 (MAX15301) | 1.80 V | FMC adjustable VCCO |
| U74 | 0x46 | MGTAVCC | U95 (MAX20751) | 0.90 V | MGT transceiver core |
| U75 | 0x47 | MGTAVTT | U96 (MAX20751) | 1.20 V | MGT termination |

#### PS_PMBUS (I2C mux channel 1) — PS / ARM side

| INA226 Chip | I2C Addr | Rail | Regulator | Rail Voltage | Description |
|---|---|---|---|---|---|
| **U76** | 0x40 | **VCCPSINTFP** | U46 (MAX15301) | 0.85 V | **PS full-power (A53 CPUs)** |
| **U77** | 0x41 | **VCCPSINTLP** | U4 (MAX15303) | 0.85 V | PS low-power (RPU) |
| U78 | 0x42 | VCCPSAUX | U3 (MAX8869E) | 1.81 V | PS auxiliary |
| U87 | 0x43 | VCCPSPLL | U17 (MAX8869E) | 1.20 V | PS PLL |
| U85 | 0x44 | MGTRAVCC | U5 (MAX8869E) | 0.85 V | PS MGT core |
| U86 | 0x45 | MGTRAVTT | U12 (MAX8869E) | 1.81 V | PS MGT termination |
| U93 | 0x46 | VCCO_PSDDR_504 | U57 (TPS22924) | 1.20 V | PS DDR I/O (bank 504) |
| U88 | 0x47 | VCCOPS | U13 (MAX15303) | 1.80 V | PS operational supply |
| U15 | 0x4A | VCCOPS3 | U31 (MAX8869E) | 1.81 V | PS operational supply 3 |
| U92 | 0x4B | VCCPSDDRPLL | U30 (MAX8869E) | 1.81 V | PS DDR PLL |

### 4.6 Idle Power Readings (Baseline)

Captured via SSH from sysfs while no inference was running:

| Sensor | Rail | Power (mW) | Current (mA) | V_bus (mV) | Notes |
|---|---|---|---|---|---|
| u79 | VCCINT | 5,968 | 7,037 | 846 | PL core — DPU idle |
| u76 | VCCPSINTFP | 1,125 | 1,326 | 848 | PS APU — ARM cores |
| u77 | VCCPSINTLP | 637 | 755 | 848 | PS RPU |
| u80 | VCCAUX | 350 | 191 | 1,796 | PL auxiliary |
| u93 | VCCO_PSDDR_504 | 225 | 183 | 1,192 | DDR I/O |
| u92 | VCCPSDDRPLL | 37 | 23 | 1,818 | DDR PLL |

**Key observation**: Even at idle, VCCINT draws ~6 W because the DPU bitstream is
loaded and the DPU cores are clocked.

**Absolute vs. dynamic power**: The benchmark script reports **absolute** average
power for each rail during the measurement window — it does **not** automatically
subtract idle baseline. To isolate DPU dynamic power for a paper, you manually
compute:

$$
P_\text{dynamic} = P_\text{inference} - P_\text{idle}
$$

The `--idle-baseline` option (see Section 5.4) captures idle power for N seconds
without running any inference, so both numbers appear in the same JSON file for
easy comparison.

### 4.7 Power Groupings for Analysis

The benchmark script aggregates power into semantic groups:

| Group | Rails | Rationale |
|---|---|---|
| **DPU_fabric** | VCCINT, VCCBRAM | Directly driven by DPU computation |
| **PS_compute** | VCCPSINTFP, VCCPSINTLP | ARM A53 cores (entropy coding, pre/post-processing) |
| **PL_total** | All 8 PL_PMBUS rails | Total PL-side power |
| **PS_total** | All 10 PS_PMBUS rails | Total PS-side power |
| **board_total** | All 18 rails | Complete board power (monitored portion) |

**Note**: The INA226 sensors do **not** cover all power rails on the board.
Unmonitored rails include DDR DRAM cells, SD card, HDMI/DP, USB, etc. The
`board_total` from INA226 is therefore a **lower bound** of actual total board
power.

**Cross-verification options**:
1. **External power meter** (recommended for publication): Measure at the 12 V
   barrel jack input with an inline meter (e.g., J7-C USB meter or bench supply
   with current readout). This gives true total board power including all
   unmonitored peripherals and regulator conversion losses. Compare:
   INA226 sum ≈ X W vs. external meter ≈ Y W → difference = unmonitored load.
2. **Maxim regulator telemetry**: The voltage regulators on MAXIM_PMBUS (mux
   channel 0) also report power, but they are not exposed via Linux hwmon and
   would require raw `i2cget` commands — fragile and not recommended.

### 4.8 Measurement Methodology

- **Polling rate**: 50 Hz (configurable via `--power-hz`)
- **Busy-wait**: The sampling thread uses `time.perf_counter()` busy-wait for
  sub-millisecond precision (avoids `time.sleep()` jitter)
- **Baseline capture**: Use `--idle-baseline <seconds>` to automatically sample
  idle power before the workload starts. Both idle and load readings appear in
  the output JSON under `power.idle_baseline` and `power.per_rail` respectively
- **Energy**: Computed as $E = P_\text{avg} \times t_\text{duration}$

---

## 5. Benchmark Script: `scripts/fpga/benchmark_fpga.py`

### 5.1 Design Principles

1. **Standardised JSON output** — all results in a machine-parseable format,
   compatible with a future GPU benchmark for direct comparison.
2. **Per-component timing** — every step (each DPU call, each entropy operation)
   is individually timed using `time.perf_counter()`.
3. **Step labels** — prefixed with `dpu_` or `cpu_` to enable automatic
   DPU vs CPU aggregation.
4. **Warmup phase** — configurable warmup iterations to reach thermal/cache steady
   state before measurement.
5. **Reproducibility** — logs metadata (timestamp, iteration counts, hardware info).

### 5.2 Scenarios

| Scenario | Steps Timed | Purpose |
|---|---|---|
| `full` | All (compress + decompress) | End-to-end latency for one tile |
| `compress` | g_a → h_a → EB → h_s → GC | Encode-only latency |
| `decompress` | EB → h_s → GC → g_s | Decode-only latency (from cached bitstream) |
| `dpu_only` | g_a, h_a, h_s, g_s | Isolate DPU latency, no entropy coding |
| `entropy_only` | EB.compress, EB.decompress, GC.compress, GC.decompress | Isolate CPU entropy coding |

### 5.3 Deployment to Board

The benchmark script is automatically included in the deploy pipeline
(`deploy.py` Phase 1 copies it into the compiled model directory). After
`deploy.py --phase transfer`, it will be at:

```
/home/root/SAR_DDC/active_model/benchmark_fpga.py
```

For manual transfer:
```bash
scp scripts/fpga/benchmark_fpga.py ZCU102:/home/root/SAR_DDC/active_model/
```

### 5.4 Usage

```bash
# On the ZCU102 board
cd /home/root/SAR_DDC/active_model

# Full pipeline with power + idle baseline + HW metadata
# Output auto-saved to results/benchmark_full.json
python3 benchmark_fpga.py \
    --xmodel ResidualScaleHyperpriorDPUWrapper_pt.xmodel \
    --scenario full \
    --warmup 20 \
    --iters 100 \
    --power \
    --power-hz 50 \
    --idle-baseline 10 \
    --collect-hw-meta

# Compress-only → results/benchmark_compress.json
python3 benchmark_fpga.py \
    --xmodel ResidualScaleHyperpriorDPUWrapper_pt.xmodel \
    --scenario compress \
    --warmup 20 --iters 100 --power

# DPU-only for isolating DPU latency → results/benchmark_dpu_only.json
python3 benchmark_fpga.py \
    --xmodel ResidualScaleHyperpriorDPUWrapper_pt.xmodel \
    --scenario dpu_only \
    --iters 200

# Entropy-only for CPU-bound analysis → results/benchmark_entropy_only.json
python3 benchmark_fpga.py \
    --xmodel ResidualScaleHyperpriorDPUWrapper_pt.xmodel \
    --scenario entropy_only \
    --iters 200
```

**Note**: When `--output` is omitted, the script automatically saves to
`results/benchmark_<scenario>.json`. Only use `--output` to override the path.

### 5.5 JSON Output Schema

```json
{
  "platform": "FPGA_ZCU102",
  "scenario": "full",
  "timestamp": "2025-01-15T...",
  "n_warmup": 20,
  "n_iters": 100,

  "latency_breakdown": {
    "dpu_g_a": {"mean_s": 0.072, "std_s": 0.001, "min_s": 0.070, "max_s": 0.075},
    "cpu_eb_compress": {"mean_s": ..., ...},
    "...": "..."
  },
  "latency_total_mean_s": 0.180,
  "latency_total_mean_ms": 180.0,
  "latency_dpu_total_mean_ms": 73.0,
  "latency_cpu_total_mean_ms": 107.0,
  "latency_wall_total_s": 18.0,

  "throughput_fps": 5.56,
  "avg_compressed_bytes": 1234,

  "power": {
    "per_rail": {
      "VCCINT": {"avg_power_w": 8.5, "energy_j": 153.0, "n_samples": 900, "duration_s": 18.0},
      "VCCPSINTFP": {"avg_power_w": 1.8, ...},
      "..."
    },
    "groups_avg_w": {
      "PL_total": 10.2,
      "PS_total": 3.5,
      "DPU_fabric": 9.8,
      "PS_compute": 2.1
    },
    "board_total_avg_w": 13.7
  },

  "hw_dpu_info": { "n_dpu_cores": 3, "dpu_freq_mhz": 300, "..." },
  "hw_xmodel_meta": { "total_workload_ops": 78338007104, "..." }
}
```

### 5.6 Key Implementation Details

#### DPUSubgraphRunner
- Wraps VART `Runner` with float-in/float-out interface
- Handles INT8 quantization (input: `× 2^fix_point`) and dequantization
  (output: `× 2^(-fix_point)`) automatically
- Uses `execute_async()` + `wait()` for single-threaded sequential execution

#### INA226PowerSampler
- Background thread with busy-wait polling (avoids `time.sleep` jitter)
- Reads `power1_input` from sysfs (micro-watts, hardware-computed by INA226)
- Returns per-rail average power (W) and total energy (J) over the measurement window
- Automatic sensor discovery: scans `/sys/class/hwmon/` for `ina226_*` names

#### StepTimer
- Uses `time.perf_counter()` for high-resolution wall-clock timing
- Mark/commit pattern: `mark("label")` records transition points,
  `commit()` saves the inter-mark deltas for one iteration
- `summary()` returns mean/std/min/max per step across all committed iterations

---

## 6. Computational Throughput Analysis

### 6.1 DPU Computational Throughput

$$
\text{Throughput}_\text{subgraph} = \text{Workload (OPs)} \times \text{FPS}
$$

| Subgraph | Workload (GOPs) | FPS | Throughput (TOPS) | DPU Utilization (%) |
|---|---|---|---|---|
| g_a | 39.75 | 27.8 | 1.105 | 89.9% |
| h_a | 0.275 | 1,225 | 0.000337 | 0.03% |
| h_s | 0.176 | 1,375 | 0.000242 | 0.02% |
| g_s | 38.13 | 28.5 | 1.087 | 88.5% |

The low utilization for h_a/h_s is not a DPU efficiency issue — it reflects the
inherently small workload due to tiny spatial dimensions (2×2 feature maps).

### 6.2 Measured Per-Tile Latency (Full Scenario)

Actual measurements from `benchmark_fpga.py --scenario full` with 100 iterations
on the ZCU102 (random input data, single thread):

| Step | Device | Latency (ms) | Std (ms) | % of Total |
|---|---|---|---|---|
| g_a (real + imag) | DPU | 79.56 | 0.10 | 18.7% |
| concat + abs | CPU | 1.16 | 0.05 | 0.3% |
| h_a | DPU | 2.16 | 0.04 | 0.5% |
| EB compress | CPU | 4.98 | 0.15 | 1.2% |
| EB decompress | CPU | 5.07 | 0.10 | 1.2% |
| h_s | DPU | 1.97 | 0.06 | 0.5% |
| **GC compress** | **CPU** | **113.35** | **0.22** | **26.7%** |
| **GC decompress** | **CPU** | **133.54** | **0.19** | **31.4%** |
| split y_hat | CPU | 0.04 | 0.00 | 0.0% |
| g_s (real + imag) | DPU | 83.15 | 0.23 | 19.6% |
| **Total** | | **425.0** | | **100%** |
| DPU subtotal | | 166.8 | | 39.3% |
| CPU subtotal | | 258.1 | | 60.7% |

**Key finding**: CPU-side Gaussian Conditional entropy coding dominates at
**58.1%** of total latency (113 + 134 = 247 ms). The DPU is not the bottleneck.
Optimising the rANS C++ codec on ARM A53 or offloading entropy coding to a
hardware accelerator would yield the largest speedup.

### 6.3 Power Summary (Full Scenario)

| Group | Power (W) |
|---|---|
| Board total (18 rails) | 11.62 |
| PL total | 9.43 |
| PS total | 2.19 |
| DPU fabric (VCCINT + VCCBRAM) | 8.98 |
| PS compute (A53 cores) | 1.77 |

### 6.4 Comparison: xdputil Synthetic vs. Real-World DPU Timing

The `dpu_g_a` step times both g_a calls (real + imag) under one mark:

| Subgraph | xdputil Synthetic (ms) | Real-World per Call (ms) | Overhead |
|---|---|---|---|
| g_a | 36.0 | ~39.8 (79.6 / 2) | +10.6% |
| h_a | 0.8 | 2.2 | +175% |
| h_s | 0.7 | 2.0 | +186% |
| g_s | 35.1 | ~41.6 (83.1 / 2) | +18.5% |

The overhead comes from: Python function call wrapping, INT8 quantisation /
dequantisation (numpy multiply + cast), VART `execute_async` + `wait` dispatch
overhead, and numpy buffer allocation. For the tiny h_a/h_s subgraphs, this
fixed overhead dominates the actual DPU computation time.

---

## 7. Derived Metrics for Publication

The following metrics can be computed from the benchmark output:

### 7.1 Latency & Throughput
- **Per-tile latency** (ms): `latency_total_mean_ms`
- **Throughput** (tiles/s): `throughput_fps`
- **DPU fraction**: `latency_dpu_total_mean_ms / latency_total_mean_ms`

### 7.2 Model Size
- **FP32 parameters**: ~15M params × 4 bytes = ~60 MB
- **INT8 quantized weights**: 14.4 MB (from xmodel CONST regions)
- **Compression ratio**: 60 / 14.4 ≈ 4.2× (quantization + pruning)

### 7.3 Computational Efficiency
- **GOPS/W** (DPU): Throughput_TOPS / DPU_fabric_power_W
- **GOPS/W** (Board): Throughput_TOPS / board_total_power_W

### 7.4 Energy per Inference
- **DPU energy/tile**: $E = P_\text{DPU\_fabric} \times t_\text{DPU\_total}$
- **Board energy/tile**: $E = P_\text{board\_total} \times t_\text{total}$

### 7.5 Compression Performance (codec metrics)
- **Bitrate** (bpp): `avg_compressed_bytes × 8 / (256 × 256)`
- **Bits per pixel** for the compressed representation

---

## 8. Known Limitations & Caveats

### 8.1 Power Measurement
1. **INA226 temporal resolution**: The INA226 integrates current over a configurable
   window (typically 1–4 ms). At 50 Hz polling we undersample relative to the
   sensor's integration time, which is acceptable for steady-state workloads but
   means we cannot capture sub-ms power transients.
2. **Incomplete board coverage**: INA226 monitors cover 18 rails but not all power
   delivery (DDR DRAM, USB, SD, DisplayPort). Board-total from INA226 is a lower
   bound.
3. **No baseline subtraction**: Idle power (~6W on VCCINT alone) includes DPU static
   power, clock trees, etc. Dynamic power should be estimated as
   $P_\text{dynamic} = P_\text{load} - P_\text{idle}$.

### 8.2 Timing
1. **Python overhead**: Using Python `time.perf_counter()` includes the Python
   function call overhead (~µs per call). For sub-ms operations (h_a, h_s), this
   overhead is non-trivial. Consider using `vaitrace` for DPU-level timing.
2. **Single-threaded DPU**: The benchmark uses 1 thread on 1 DPU core. The board
   has 3 DPU cores; multi-threaded operation could improve throughput by ~3× but
   increases complexity.
3. **Entropy coding is sequential**: The C++ rANS entropy coding (via `ans` module)
   runs on the ARM A53, which is relatively slow. This may dominate total latency.

### 8.3 Comparison Fairness (GPU vs FPGA)
1. **Data format**: FPGA uses INT8, GPU uses FP32 — the FPGA's lower precision
   introduces quantization error. Quality comparison (PSNR, SSIM) is essential.
2. **Batch size**: GPU batching amortises overhead; FPGA benchmark uses batch=1.
   For fair throughput comparison, normalise to per-image values.
3. **Power comparison**: GPU power from `nvidia-smi` is board-level GPU power.
   FPGA power from INA226 is PL+PS. Neither captures host/memory system power.
   Use energy-per-inference as the fairest comparison metric.

---

## 9. GPU / CPU Benchmark: `scripts/benchmark_gpu.py`

### 9.1 Purpose & Relationship to FPGA Benchmark

`benchmark_gpu.py` measures per-component latency, throughput, and power for the
**same model** running on a CUDA GPU and/or host CPU.  It produces JSON output with
the **same schema** as `benchmark_fpga.py` so results can be loaded into a single
comparison table or plot.

Key differences from the FPGA benchmark:
- **Precision**: FP32 on GPU/CPU vs. INT8 on FPGA — quality (PSNR/SSIM) **must**
  be compared alongside speed.
- **Batch size**: Always 1 (matching the FPGA baseline).
- **Entropy coding**: Same CompressAI Python implementation runs on **both** GPU
  and CPU host.  On the FPGA, this runs on the ARM A53 via C++ `ans.so`.

### 9.2 Dual-Device Mode

By default, a single invocation measures on **GPU first, then CPU sequentially**.
Skip either with:
- `--no-gpu` — skip GPU measurement (useful on CPU-only machines)
- `--no-cpu` — skip CPU measurement (faster iteration on GPU numbers)

Each device produces its own JSON file:
```
results/benchmark/<run_name>/benchmark_gpu_<scenario>.json
results/benchmark/<run_name>/benchmark_cpu_<scenario>.json
```

### 9.3 Scenarios

| Scenario | Steps Timed | Purpose |
|---|---|---|
| `full` | All (compress + decompress) | End-to-end latency for one tile |
| `compress` | g_a → h_a → EB → h_s → GC | Encode-only latency |
| `decompress` | EB → h_s → GC → g_s | Decode-only latency (from cached bitstream) |
| `nn_only` | g_a, h_a, h_s, g_s | Isolate NN latency, no entropy coding |
| `entropy_only` | EB + GC (compress + decompress) | Isolate CPU entropy coding |

The `nn_only` scenario is analogous to `dpu_only` on the FPGA.

### 9.4 Step Labels & Prefixing Convention

| Prefix | Meaning | When used |
|---|---|---|
| `gpu_` | NN subgraph running on GPU | GPU measurement mode |
| `nn_` | NN subgraph running on CPU | CPU measurement mode |
| `cpu_` | CPU-side operation (entropy coding, concat, split) | Both modes |

This allows automatic aggregation (e.g., sum all `gpu_*` steps for total GPU NN time).

### 9.5 Timing Methodology

| Device | Method | Precision |
|---|---|---|
| **GPU** (wall-clock) | `time.perf_counter()` with `torch.cuda.synchronize()` | ~µs |
| **GPU** (CUDA events) | `torch.cuda.Event(enable_timing=True)` | ~µs, no CPU-side jitter |
| **CPU** | `time.perf_counter()` | ~µs |

The GPU measurement records **both** wall-clock and CUDA event timings for every
step.  CUDA events are stored in `latency_breakdown_cuda_events` — prefer these
for NN sub-graph comparisons as they exclude Python/CPU overhead.

### 9.6 Power Measurement

| Source | Metric | How |
|---|---|---|
| **GPU** | Board-level GPU draw | `nvidia-smi --query-gpu=power.draw` polled at `--power-hz` (default 10 Hz) in a background thread |
| **CPU** | Package + DRAM power | Intel RAPL via `/sys/class/powercap/intel-rapl/` — energy counter delta between start/stop |

**Limitations**:
- `nvidia-smi` power is the **full GPU board** (incl. idle), not incremental.
- RAPL reports **package** (all cores + uncore) and **DRAM**, but not
  motherboard, PSU, fans, etc.
- Neither captures host system total power.  For publication, consider an
  external wall-plug meter.

### 9.7 Usage

The script accepts the model either via `--model-dir` (recommended — reads the
checkpoint path from `manifest.json`) or `--ckpt` (direct checkpoint path).
The two are mutually exclusive.

```bash
# ---- Recommended: use --model-dir (reads manifest.json → checkpoint) ----

# Full pipeline, GPU + CPU, with power measurement
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario full \
    --warmup 20 --iters 100 \
    --power

# GPU only, compress scenario, idle baseline
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario compress \
    --no-cpu --power --idle-baseline 10

# CPU only, entropy isolation
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario entropy_only \
    --no-gpu --iters 200

# ---- Alternative: direct checkpoint path ----

python scripts/benchmark_gpu.py \
    --ckpt logs/train/runs/<run>/checkpoints/last.ckpt \
    --scenario full \
    --warmup 20 --iters 100

# Custom output directory (works with either source)
python scripts/benchmark_gpu.py \
    --model-dir results/fpga/active_model/ \
    --scenario full \
    --output-dir results/my_benchmark
```

When `--model-dir` is used, the output directory defaults to
`results/benchmark/<model_name>/` (e.g. `ResSHyp-relu_s1_L1000_pt`).
When `--ckpt` is used, it defaults to `results/benchmark/<run_timestamp>/`.

### 9.8 JSON Output Schema

```json
{
  "platform": "GPU_NVIDIA_RTX_A4000",
  "device": "cuda",
  "scenario": "full",
  "timestamp": "2025-...",
  "n_warmup": 20,
  "n_iters": 100,

  "latency_breakdown": {
    "preprocess":       {"mean_s": 0.000001, "std_s": ..., "median_s": ..., "min_s": ..., "max_s": ..., "p95_s": ..., "n": 100},
    "gpu_g_a":          {"mean_s": 0.0012,   ...},
    "cpu_concat_abs":   {"mean_s": 0.00003,  ...},
    "gpu_h_a":          {"mean_s": 0.0005,   ...},
    "cpu_eb_compress":  {"mean_s": 0.035,    ...},
    "cpu_eb_decompress":{"mean_s": 0.002,    ...},
    "gpu_h_s":          {"mean_s": 0.0005,   ...},
    "cpu_gc_compress":  {"mean_s": 0.12,     ...},
    "cpu_gc_decompress":{"mean_s": 0.14,     ...},
    "cpu_split_y_hat":  {"mean_s": 0.000002, ...},
    "gpu_g_s":          {"mean_s": 0.0014,   ...},
    "postprocess":      {"mean_s": 0.000001, ...}
  },
  "latency_breakdown_cuda_events": {
    "preprocess":       {"mean_s": ..., ...},
    "gpu_g_a":          {"mean_s": 0.00098, ...},
    "..."
  },

  "latency_total_mean_s": 0.301,
  "latency_total_mean_ms": 301.0,
  "latency_gpu_total_mean_ms": 3.6,
  "latency_cpu_total_mean_ms": 297.4,
  "latency_wall_total_s": 30.1,

  "throughput_fps": 3.32,
  "avg_compressed_bytes": 1234,

  "power": {
    "gpu":            {"avg_power_w": 85.0, "energy_j": 2550.0, "n_samples": 300, "duration_s": 30.1},
    "cpu_rapl": {
      "package-0":    {"avg_power_w": 42.0, "energy_j": 1264.2, ...},
      "dram":         {"avg_power_w": 5.2,  "energy_j": 156.5, ...}
    },
    "gpu_avg_w":           85.0,
    "cpu_rapl_total_avg_w": 47.2,
    "idle_gpu":            {"avg_power_w": 15.0, ...},
    "idle_cpu_rapl": {
      "package-0":    {"avg_power_w": 12.0, ...},
      "dram":         {"avg_power_w": 3.0,  ...}
    }
  },

  "hw_info": {
    "name": "NVIDIA RTX A4000",
    "cuda_version": "12.4",
    "cudnn_version": "90100",
    "torch_version": "2.5.1+cu124",
    "memory_total_mb": 16376,
    "power_limit_w": 140.0,
    "max_sm_clock_mhz": 1560.0,
    "max_mem_clock_mhz": 7001.0
  },

  "model_info": {
    "total_params": 15000000,
    "trainable_params": 15000000,
    "weights_size_mb": 57.22,
    "N": 128,
    "M": 256
  }
}
```

### 9.9 Interpreting Results & Cross-Platform Comparison

#### 9.9.1 What to compare

| Metric | Fair comparison? | Notes |
|---|---|---|
| **NN latency** (gpu/dpu/nn steps) | ✅ Comparable | Different devices executing the same subgraphs |
| **Entropy latency** (cpu_ steps) | ⚠️ Be careful | GPU benchmark runs entropy on x86; FPGA on ARM A53. The x86 is vastly faster |
| **Total latency** | ✅ Comparable | Apples-to-apples if batch=1 |
| **Throughput** (fps) | ✅ Comparable | Derived from total latency |
| **Power** | ⚠️ Different scopes | GPU = board GPU power; FPGA = SoC rails. Not directly comparable |
| **Energy per inference** | ✅ Best metric | $E = P_\text{avg} \times t_\text{total}$ for each platform |
| **Quality** (PSNR/SSIM) | ✅ Essential | FP32 vs INT8 quality gap must be reported alongside speed |

#### 9.9.2 Pitfalls

1. **Entropy coding dominates on all platforms.**  On both GPU and FPGA, entropy
   coding (CompressAI's rANS) runs on the CPU.  The x86 host is ~10–30× faster
   than the ARM A53, so total latency differences are dominated by this component
   rather than NN inference speed.
2. **GPU warmup.**  The first CUDA kernel launch incurs JIT compilation overhead.
   Always use `--warmup ≥ 10` to ensure steady-state.
3. **GPU power states.**  An idle GPU draws ~15W.  Under load, RTX A4000 can reach
   ~140W.  If `--iters` is too low, the GPU may not reach steady-state power,
   inflating apparent energy efficiency.
4. **CUDA events vs. wall-clock.**  `latency_breakdown_cuda_events` excludes
   CPU-side overhead (Python dispatch, memory copies).  For GPU NN steps, CUDA
   events are more accurate; for end-to-end latency, use the wall-clock breakdown.
5. **CPU benchmark is single-threaded by default.**  PyTorch uses `torch.get_num_threads()`
   threads for CPU ops.  This may differ across machines.  Report `torch_threads`
   from the JSON output alongside results.

#### 9.9.3 Recommended comparison table format

For a publication, present results as:

| | FPGA (ZCU102) | GPU (RTX A4000) | CPU (host) |
|---|---|---|---|
| **NN latency** (ms) | 167 | ? | ? |
| **Entropy latency** (ms) | 258 | ? | ? |
| **Total latency** (ms) | 425 | ? | ? |
| **Throughput** (patches/s) | 2.35 | ? | ? |
| **Power** (W) | 11.62 (SoC) | ? (GPU board) | ? (RAPL pkg) |
| **Energy/patch** (J) | 4.94 | ? | ? |
| **PSNR** (dB) | from eval | from eval | from eval |
| **Precision** | INT8 | FP32 | FP32 |

---

## 10. Future Work

- [x] **GPU benchmark** (`benchmark_gpu.py`): Mirror the same JSON schema with
  `torch.cuda.Event` timing, NVIDIA power readings, and Intel RAPL.
- [ ] **Run GPU benchmark and populate Section 9 with real results**.
- [ ] **vaitrace validation**: Run `vaitrace` once to get DPU-level hardware timing
  and validate against our Python-level measurements.
- [ ] **Tile-level benchmarking**: Benchmark on full 512×512 or 1024×1024 images
  using `patch_infer_fpga()` with overlap-blended tiling.
- [ ] **Multi-thread DPU**: Test 3-thread operation to measure throughput scaling.
- [ ] **USB power meter**: External 12V input measurement for true board power.
- [ ] **Thermal characterization**: Monitor DPU junction temperature during extended
  runs to assess thermal throttling risk.

---

## 11. References

- **UG1182** (v1.7): Xilinx ZCU102 Evaluation Board User Guide, February 2023.
  - Table 3-22: I2C0 U60 Mux Target Bus Connections (INA226 chip U-ref → I2C address).
  - Table 3-55: Power System Devices (regulator → rail → INA226 address).
  - Table 3-56: ZCU102 Power Rails with INA226 Power Monitors (subset of 3-55).
- **DS925**: Zynq UltraScale+ MPSoC Data Sheet — DC and AC Switching Characteristics.
  Defines each power rail's function in the "Recommended Operating Conditions" table.
- **UG583**: UltraScale Architecture PCB Design User Guide.
  Power domain consolidation tables mapping rails to LPD/FPD/PLPD/BPD domains.
- **UG1085**: Zynq UltraScale+ Device Technical Reference Manual.
  Defines what hardware lives in each power domain.
- **PG338**: DPUCZDX8G Product Guide (Vitis-AI DPU architecture).
- **Vitis-AI 3.0 User Guide**: DPU runtime APIs (VART), xdputil tooling.
- **TI INA226 datasheet**: High/low-side current/power monitor, 16-bit ADC.
