# DATE'27 measurement campaign — MANIFEST

Canonical result tree for the DATE'27 coherent re-run (`docs/DATE27_paper_plan.md` §4.1/§4.1a).
One line per run, appended immediately after the run finishes. Figure scripts read from this tree
only. Filenames encode the exact flag set (e.g. `fo_t16_pf_neon_ent_warm.json`).

Global setup (all runs unless noted): λ=20, seed 0, overlap 2, snap grid, full scene
(`data/full_scene_i16.npy`), warm read, power sampling on, batch 1. `make clean` full rebuild done
once at campaign start (confirmed per arch below).

**Columns:** output file · full CLI flags · host git SHA · board git SHA · model (arch/λ/seed) · date
· make-clean confirmed

| Output file | CLI flags | Host SHA | Board SHA | Model | Date | make clean |
| --- | --- | --- | --- | --- | --- | --- |
| results/date27/ladder/ResFP/r0_seq_warm.json | `--schedule seq --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r0_seq_cold.json | `--schedule seq` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 cold (SD-read baseline) |
| results/date27/ladder/ResFP/r1_p0_t4_warm.json | `--schedule p0 --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r2_fo_t3_lanemaj_warm.json | `--schedule p0 --fanout --lane-major --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r3_fo_t3_warm.json | `--schedule p0 --fanout --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r4_fo_t6_warm.json | `--schedule p0 --fanout --threads 6 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r5_fo_t6_neon_warm.json | `--schedule p0 --fanout --threads 6 --neon --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r6_fo_t6_neon_pf_warm.json | `--schedule p0 --fanout --threads 6 --neon --prefetch --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResFP/r7_fo_t6_neon_pf_ent_warm.json | `--schedule p0 --fanout --threads 6 --neon --prefetch --entropy --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/lanes/ResFP/t1_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 1 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=1 |
| results/date27/lanes/ResFP/t1_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=1 |
| results/date27/lanes/ResFP/t2_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 2 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=2 |
| results/date27/lanes/ResFP/t2_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=2 |
| results/date27/lanes/ResFP/t3_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=3 |
| results/date27/lanes/ResFP/t3_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=3 |
| results/date27/lanes/ResFP/t4_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=4 |
| results/date27/lanes/ResFP/t4_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=4 |
| results/date27/lanes/ResFP/t5_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 5 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=5 |
| results/date27/lanes/ResFP/t5_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=5 |
| results/date27/lanes/ResFP/t6_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 6 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=6 |
| results/date27/lanes/ResFP/t6_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=6 |
| results/date27/lanes/ResFP/t8_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 8 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=8 |
| results/date27/lanes/ResFP/t8_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=8 |
| results/date27/lanes/ResFP/t10_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 10 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=10 |
| results/date27/lanes/ResFP/t10_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=10 |
| results/date27/lanes/ResFP/t12_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 12 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=12 |
| results/date27/lanes/ResFP/t12_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=12 |
| results/date27/lanes/ResFP/t16_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 16 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=16 |
| results/date27/lanes/ResFP/t16_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=16 |
| results/date27/lanes/ResFP/t20_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 20 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=20 |
| results/date27/lanes/ResFP/t20_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=20 |
| results/date27/lanes/ResFP/t24_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 24 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=24 |
| results/date27/lanes/ResFP/t24_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=24 |
| results/date27/lanes/ResFP/t32_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 32 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=32 |
| results/date27/lanes/ResFP/t32_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=32 |
| results/date27/lanes/ResFP/t48_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 48 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=48 |
| results/date27/lanes/ResFP/t48_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=48 |
| results/date27/lanes/ResFP/t64_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 64 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=64 |
| results/date27/lanes/ResFP/t64_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=64 |
| results/date27/s0/ResFP/s0_compress_entoff.json | `--config s0 --scenario compress --no-entropy-opt --power` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E3 |
| results/date27/s0/ResFP/ResFP-relu_s0_L20_pt_xmodel_info.json | `collect_roofline.py` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E3 xdputil peaks |
| results/date27/vaitrace/ResFP/vaitrace_1lane.txt | `vaitrace --txt_summary --fanout --threads 1 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/vaitrace/ResFP/vaitrace_knee6.txt | `vaitrace --txt_summary --fanout --threads 6 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/occupancy/ResFP/r4_full.csv | `--schedule p0 --fanout --threads 6 --keep-cache --trace date27_scratch/occ_r4_full.csv` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E5 r4 |
| results/date27/occupancy/ResFP/r7_full.csv | `--schedule p0 --fanout --threads 6 --neon --prefetch --entropy --keep-cache --trace date27_scratch/occ_r7_full.csv` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E5 r7 |
| results/date27/checks/ResFP/sha256_r0_vs_r7.txt | `sha256sum r0.ddc r7.ddc` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E6 byte-identity gate |
| results/date27/checks/ResFP/sd_cold_read.json | `(derived from E1 r0_seq cold)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E6 SD cold-read |
| results/date27/ladder/FP/r0_seq_warm.json | `--schedule seq --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r0_seq_cold.json | `--schedule seq` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 cold (SD-read baseline) |
| results/date27/ladder/FP/r1_p0_t4_warm.json | `--schedule p0 --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r2_fo_t3_lanemaj_warm.json | `--schedule p0 --fanout --lane-major --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r3_fo_t3_warm.json | `--schedule p0 --fanout --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r4_fo_t32_warm.json | `--schedule p0 --fanout --threads 32 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r5_fo_t32_neon_warm.json | `--schedule p0 --fanout --threads 32 --neon --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r6_fo_t32_neon_pf_warm.json | `--schedule p0 --fanout --threads 32 --neon --prefetch --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/FP/r7_fo_t32_neon_pf_ent_warm.json | `--schedule p0 --fanout --threads 32 --neon --prefetch --entropy --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/lanes/FP/t1_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 1 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=1 |
| results/date27/lanes/FP/t1_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=1 |
| results/date27/lanes/FP/t2_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 2 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=2 |
| results/date27/lanes/FP/t2_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=2 |
| results/date27/lanes/FP/t3_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=3 |
| results/date27/lanes/FP/t3_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=3 |
| results/date27/lanes/FP/t4_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=4 |
| results/date27/lanes/FP/t4_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=4 |
| results/date27/lanes/FP/t5_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 5 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=5 |
| results/date27/lanes/FP/t5_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=5 |
| results/date27/lanes/FP/t6_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 6 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=6 |
| results/date27/lanes/FP/t6_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=6 |
| results/date27/lanes/FP/t8_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 8 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=8 |
| results/date27/lanes/FP/t8_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=8 |
| results/date27/lanes/FP/t10_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 10 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=10 |
| results/date27/lanes/FP/t10_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=10 |
| results/date27/lanes/FP/t12_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 12 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=12 |
| results/date27/lanes/FP/t12_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=12 |
| results/date27/lanes/FP/t16_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 16 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=16 |
| results/date27/lanes/FP/t16_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=16 |
| results/date27/lanes/FP/t20_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 20 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=20 |
| results/date27/lanes/FP/t20_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=20 |
| results/date27/lanes/FP/t24_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 24 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=24 |
| results/date27/lanes/FP/t24_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=24 |
| results/date27/lanes/FP/t32_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 32 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=32 |
| results/date27/lanes/FP/t32_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=32 |
| results/date27/lanes/FP/t48_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 48 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=48 |
| results/date27/lanes/FP/t48_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=48 |
| results/date27/lanes/FP/t64_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 64 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=64 |
| results/date27/lanes/FP/t64_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=64 |
| results/date27/s0/FP/s0_compress_entoff.json | `--config s0 --scenario compress --no-entropy-opt --power` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E3 |
| results/date27/s0/FP/FP-relu_s0_L20_pt_xmodel_info.json | `collect_roofline.py` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E3 xdputil peaks |
| results/date27/vaitrace/FP/vaitrace_1lane.txt | `vaitrace --txt_summary --fanout --threads 1 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/vaitrace/FP/vaitrace_knee32.txt | `vaitrace --txt_summary --fanout --threads 32 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/occupancy/FP/r4_full.csv | `--schedule p0 --fanout --threads 32 --keep-cache --trace date27_scratch/occ_r4_full.csv` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E5 r4 |
| results/date27/occupancy/FP/r7_full.csv | `--schedule p0 --fanout --threads 32 --neon --prefetch --entropy --keep-cache --trace date27_scratch/occ_r7_full.csv` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E5 r7 |
| results/date27/checks/FP/sha256_r0_vs_r7.txt | `sha256sum r0.ddc r7.ddc` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E6 byte-identity gate |
| results/date27/checks/FP/sd_cold_read.json | `(derived from E1 r0_seq cold)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E6 SD cold-read |
| results/date27/ladder/SHyp/r0_seq_warm.json | `--schedule seq --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r0_seq_cold.json | `--schedule seq` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 cold (SD-read baseline) |
| results/date27/ladder/SHyp/r1_p0_t4_warm.json | `--schedule p0 --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r2_fo_t3_lanemaj_warm.json | `--schedule p0 --fanout --lane-major --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r3_fo_t3_warm.json | `--schedule p0 --fanout --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r4_fo_t24_warm.json | `--schedule p0 --fanout --threads 24 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r5_fo_t24_neon_warm.json | `--schedule p0 --fanout --threads 24 --neon --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r6_fo_t24_neon_pf_warm.json | `--schedule p0 --fanout --threads 24 --neon --prefetch --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/SHyp/r7_fo_t24_neon_pf_ent_warm.json | `--schedule p0 --fanout --threads 24 --neon --prefetch --entropy --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/lanes/SHyp/t1_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 1 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=1 |
| results/date27/lanes/SHyp/t1_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=1 |
| results/date27/lanes/SHyp/t2_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 2 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=2 |
| results/date27/lanes/SHyp/t2_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=2 |
| results/date27/lanes/SHyp/t3_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=3 |
| results/date27/lanes/SHyp/t3_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=3 |
| results/date27/lanes/SHyp/t4_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=4 |
| results/date27/lanes/SHyp/t4_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=4 |
| results/date27/lanes/SHyp/t5_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 5 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=5 |
| results/date27/lanes/SHyp/t5_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=5 |
| results/date27/lanes/SHyp/t6_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 6 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=6 |
| results/date27/lanes/SHyp/t6_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=6 |
| results/date27/lanes/SHyp/t8_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 8 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=8 |
| results/date27/lanes/SHyp/t8_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=8 |
| results/date27/lanes/SHyp/t10_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 10 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=10 |
| results/date27/lanes/SHyp/t10_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=10 |
| results/date27/lanes/SHyp/t12_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 12 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=12 |
| results/date27/lanes/SHyp/t12_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=12 |
| results/date27/lanes/SHyp/t16_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 16 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=16 |
| results/date27/lanes/SHyp/t16_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=16 |
| results/date27/lanes/SHyp/t20_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 20 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=20 |
| results/date27/lanes/SHyp/t20_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=20 |
| results/date27/lanes/SHyp/t24_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 24 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=24 |
| results/date27/lanes/SHyp/t24_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=24 |
| results/date27/lanes/SHyp/t32_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 32 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=32 |
| results/date27/lanes/SHyp/t32_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=32 |
| results/date27/lanes/SHyp/t48_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 48 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=48 |
| results/date27/lanes/SHyp/t48_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=48 |
| results/date27/lanes/SHyp/t64_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 64 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=64 |
| results/date27/lanes/SHyp/t64_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=64 |
| results/date27/s0/SHyp/s0_compress_entoff.json | `--config s0 --scenario compress --no-entropy-opt --power` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E3 |
| results/date27/s0/SHyp/SHyp-relu_s0_L20_pt_xmodel_info.json | `collect_roofline.py` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E3 xdputil peaks |
| results/date27/vaitrace/SHyp/vaitrace_1lane.txt | `vaitrace --txt_summary --fanout --threads 1 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/vaitrace/SHyp/vaitrace_knee24.txt | `vaitrace --txt_summary --fanout --threads 24 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/occupancy/SHyp/r4_full.csv | `--schedule p0 --fanout --threads 24 --keep-cache --trace date27_scratch/occ_r4_full.csv` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E5 r4 |
| results/date27/occupancy/SHyp/r7_full.csv | `--schedule p0 --fanout --threads 24 --neon --prefetch --entropy --keep-cache --trace date27_scratch/occ_r7_full.csv` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E5 r7 |
| results/date27/checks/SHyp/sha256_r0_vs_r7.txt | `sha256sum r0.ddc r7.ddc` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 byte-identity gate |
| results/date27/checks/SHyp/sd_cold_read.json | `(derived from E1 r0_seq cold)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 SD cold-read |
| results/date27/ladder/ResSHyp/r0_seq_warm.json | `--schedule seq --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r0_seq_cold.json | `--schedule seq` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 cold (SD-read baseline) |
| results/date27/ladder/ResSHyp/r1_p0_t4_warm.json | `--schedule p0 --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r2_fo_t3_lanemaj_warm.json | `--schedule p0 --fanout --lane-major --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r3_fo_t3_warm.json | `--schedule p0 --fanout --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r4_fo_t20_warm.json | `--schedule p0 --fanout --threads 20 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r5_fo_t20_neon_warm.json | `--schedule p0 --fanout --threads 20 --neon --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r6_fo_t20_neon_pf_warm.json | `--schedule p0 --fanout --threads 20 --neon --prefetch --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/ladder/ResSHyp/r7_fo_t20_neon_pf_ent_warm.json | `--schedule p0 --fanout --threads 20 --neon --prefetch --entropy --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E1 warm |
| results/date27/lanes/ResSHyp/t1_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 1 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=1 |
| results/date27/lanes/ResSHyp/t1_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=1 |
| results/date27/lanes/ResSHyp/t2_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 2 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=2 |
| results/date27/lanes/ResSHyp/t2_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=2 |
| results/date27/lanes/ResSHyp/t3_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 3 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=3 |
| results/date27/lanes/ResSHyp/t3_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=3 |
| results/date27/lanes/ResSHyp/t4_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 4 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=4 |
| results/date27/lanes/ResSHyp/t4_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=4 |
| results/date27/lanes/ResSHyp/t5_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 5 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=5 |
| results/date27/lanes/ResSHyp/t5_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=5 |
| results/date27/lanes/ResSHyp/t6_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 6 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=6 |
| results/date27/lanes/ResSHyp/t6_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=6 |
| results/date27/lanes/ResSHyp/t8_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 8 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=8 |
| results/date27/lanes/ResSHyp/t8_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=8 |
| results/date27/lanes/ResSHyp/t10_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 10 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=10 |
| results/date27/lanes/ResSHyp/t10_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=10 |
| results/date27/lanes/ResSHyp/t12_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 12 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=12 |
| results/date27/lanes/ResSHyp/t12_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=12 |
| results/date27/lanes/ResSHyp/t16_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 16 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=16 |
| results/date27/lanes/ResSHyp/t16_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=16 |
| results/date27/lanes/ResSHyp/t20_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 20 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=20 |
| results/date27/lanes/ResSHyp/t20_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=20 |
| results/date27/lanes/ResSHyp/t24_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 24 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=24 |
| results/date27/lanes/ResSHyp/t24_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=24 |
| results/date27/lanes/ResSHyp/t32_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 32 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=32 |
| results/date27/lanes/ResSHyp/t32_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=32 |
| results/date27/lanes/ResSHyp/t48_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 48 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=48 |
| results/date27/lanes/ResSHyp/t48_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=48 |
| results/date27/lanes/ResSHyp/t64_fo_neon_pf_ent_warm.json | `--schedule p0 --fanout --neon --prefetch --entropy --threads 64 --keep-cache` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 lane=64 |
| results/date27/lanes/ResSHyp/t64_occtrace.csv | `(fetched trace CSV)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E2 occ-trace lane=64 |
| results/date27/s0/ResSHyp/s0_compress_entoff.json | `--config s0 --scenario compress --no-entropy-opt --power` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E3 |
| results/date27/s0/ResSHyp/ResSHyp-relu_s0_L20_pt_xmodel_info.json | `collect_roofline.py` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E3 xdputil peaks |
| results/date27/vaitrace/ResSHyp/vaitrace_1lane.txt | `vaitrace --txt_summary --fanout --threads 1 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/vaitrace/ResSHyp/vaitrace_knee20.txt | `vaitrace --txt_summary --fanout --threads 20 --max-rows 16` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E4 |
| results/date27/occupancy/ResSHyp/r4_full.csv | `--schedule p0 --fanout --threads 20 --keep-cache --trace date27_scratch/occ_r4_full.csv` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E5 r4 |
| results/date27/occupancy/ResSHyp/r7_full.csv | `--schedule p0 --fanout --threads 20 --neon --prefetch --entropy --keep-cache --trace date27_scratch/occ_r7_full.csv` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E5 r7 |
| results/date27/checks/ResSHyp/sha256_r0_vs_r7.txt | `sha256sum r0.ddc r7.ddc` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 byte-identity gate |
| results/date27/checks/ResSHyp/sd_cold_read.json | `(derived from E1 r0_seq cold)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 SD cold-read |
| results/date27/checks/ResFP/mem_64L_raw.txt | `(/proc/<pid>/status VmHWM/VmRSS + /proc/meminfo CmaTotal/CmaFree, sampled 1 Hz during the E2 64L run)` | 3a198c3+dirty | 3a198c3+dirty | ResFP-relu_s0_L20_pt | 2026-08-31 | yes — E6 peak RSS/CMA at 64L (CORRECTED 2026-09-01: original pgrep -f pattern self-matched the polling shell, not stream_pipeline; VmHWM/VmRSS read ~2.5 MiB instead of the real process; refixed via /proc/*/exe symlink match + re-measured; CMA figures were unaffected — CmaTotal/CmaFree are system-wide, not PID-scoped) |
| results/date27/checks/FP/mem_64L_raw.txt | `(/proc/<pid>/status VmHWM/VmRSS + /proc/meminfo CmaTotal/CmaFree, sampled 1 Hz during the E2 64L run)` | 3a198c3+dirty | 3a198c3+dirty | FP-relu_s0_L20_pt | 2026-08-31 | yes — E6 peak RSS/CMA at 64L (CORRECTED 2026-09-01: original pgrep -f pattern self-matched the polling shell, not stream_pipeline; VmHWM/VmRSS read ~2.5 MiB instead of the real process; refixed via /proc/*/exe symlink match + re-measured; CMA figures were unaffected — CmaTotal/CmaFree are system-wide, not PID-scoped) |
| results/date27/checks/SHyp/mem_64L_raw.txt | `(/proc/<pid>/status VmHWM/VmRSS + /proc/meminfo CmaTotal/CmaFree, sampled 1 Hz during the E2 64L run)` | 3a198c3+dirty | 3a198c3+dirty | SHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 peak RSS/CMA at 64L (CORRECTED 2026-09-01: original pgrep -f pattern self-matched the polling shell, not stream_pipeline; VmHWM/VmRSS read ~2.5 MiB instead of the real process; refixed via /proc/*/exe symlink match + re-measured; CMA figures were unaffected — CmaTotal/CmaFree are system-wide, not PID-scoped) |
| results/date27/checks/ResSHyp/mem_64L_raw.txt | `(/proc/<pid>/status VmHWM/VmRSS + /proc/meminfo CmaTotal/CmaFree, sampled 1 Hz during the E2 64L run)` | 3a198c3+dirty | 3a198c3+dirty | ResSHyp-relu_s0_L20_pt | 2026-08-31 | yes — E6 peak RSS/CMA at 64L (CORRECTED 2026-09-01: original pgrep -f pattern self-matched the polling shell, not stream_pipeline; VmHWM/VmRSS read ~2.5 MiB instead of the real process; refixed via /proc/*/exe symlink match + re-measured; CMA figures were unaffected — CmaTotal/CmaFree are system-wide, not PID-scoped) |
| results/date27/ladder/FP/r4_fo_t12_warm.json | `--schedule p0 --fanout --threads 12 --keep-cache` | e4ea783 | e4ea783 | FP-relu_s0_L20_pt | 2026-09-01 | yes — FP knee-12 re-run r4 knee-pinned |
| results/date27/ladder/FP/r5_fo_t12_neon_warm.json | `--schedule p0 --fanout --threads 12 --neon --keep-cache` | e4ea783 | e4ea783 | FP-relu_s0_L20_pt | 2026-09-01 | yes — FP knee-12 re-run r5 +neon |
| results/date27/ladder/FP/r6_fo_t12_neon_pf_warm.json | `--schedule p0 --fanout --threads 12 --neon --prefetch --keep-cache` | e4ea783 | e4ea783 | FP-relu_s0_L20_pt | 2026-09-01 | yes — FP knee-12 re-run r6 +prefetch |
| results/date27/ladder/FP/r7_fo_t12_neon_pf_ent_warm.json | `--schedule p0 --fanout --threads 12 --neon --prefetch --entropy --keep-cache` | e4ea783 | e4ea783 | FP-relu_s0_L20_pt | 2026-09-01 | yes — FP knee-12 re-run r7 +entropy |

**P0.7 — the last board session (2026-09-01, `DATE27_paper_plan.md` §4.1 P0.7).** Build provenance:
`git diff e4ea783..HEAD -- inference_cpp/` is empty (only doc/data commits since the FP knee re-run),
so the board `stream_pipeline` binary is still the campaign build (`make clean` 2026-08-31; campaign
tree `3a198c3+dirty` ≡ `e4ea783` for `inference_cpp/`). **No rebuild** — "board SHA" below is the
binary's provenance, not a fresh build. Host SHA `7856483`. Global setup unchanged (λ=20, seed 0,
overlap 2, snap grid, full scene, warm read, power on, batch 1). Model redeployed per arch with
`deploy.py --model-name <arch>-relu_s0_L20_pt --skip-compile --skip-infer --skip-fetch`.

| Output file | CLI flags | Host SHA | Board SHA | Model | Date | make clean |
| --- | --- | --- | --- | --- | --- | --- |
| results/date27/vaitrace/FP/vaitrace_knee12.txt | `vaitrace -t 160 --txt_summary -o <out> ./build_cpp/stream_pipeline --xmodel active_model/*.xmodel --params active_model/entropy_params --tile data/full_scene_i16.npy --out /tmp/vt_knee12.ddc --windowed --p0 --fanout --threads 12 --max-rows 16 --overlap 2` (re-trace of the 12 L knee; `vaitrace_knee32.txt` kept for provenance) | 7856483 | e4ea783 (binary unchanged) | FP-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 (campaign make clean 2026-08-31) |
| results/date27/occupancy/FP/r4_full_t12.csv | `stream_benchmark.py --schedule p0 --fanout --threads 12 --keep-cache --cooldown --iters 1 --tile data/full_scene_i16.npy --trace date27_scratch/occ_r4_full_t12.csv` → `stream_pipeline --p0 --threads 12 --fanout --windowed --overlap 2 --trace …` (run JSON: `occupancy/FP/r4_trace_run_t12.json`; 32 L `r4_full.csv` kept) | 7856483 | e4ea783 (binary unchanged) | FP-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
| results/date27/occupancy/FP/r7_full_t12.csv | `stream_benchmark.py --schedule p0 --fanout --neon --prefetch --entropy --threads 12 --keep-cache --cooldown --iters 1 --tile data/full_scene_i16.npy --trace date27_scratch/occ_r7_full_t12.csv` → `stream_pipeline --p0 --threads 12 --fanout --windowed --prefetch --neon --entropy --overlap 2 --trace …` (run JSON: `occupancy/FP/r7_trace_run_t12.json`; 32 L `r7_full.csv` kept) | 7856483 | e4ea783 (binary unchanged) | FP-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
| results/date27/cpu_probe/FP/knee_12_mpstat.log | `mpstat -P ALL 1` + `pidstat -u -h -p <pid> 1` during `stream_pipeline --p0 --threads 12 --fanout --windowed --prefetch --neon --entropy --power --overlap 2` (warm: 1 warmup run first; cooled <58 °C; companions `knee_12_pidstat.log`, `knee_12_stream.log` = operating-point record) | 7856483 | e4ea783 (binary unchanged) | FP-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
| results/date27/cpu_probe/SHyp/knee_24_mpstat.log | `mpstat -P ALL 1` + `pidstat …` during `stream_pipeline --p0 --threads 24 --fanout --windowed --prefetch --neon --entropy --power --overlap 2` (warm; 1 warmup; cooled <58 °C; companions `knee_24_pidstat.log`, `knee_24_stream.log`) | 7856483 | e4ea783 (binary unchanged) | SHyp-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
| results/date27/cpu_probe/ResFP/knee_6_mpstat.log | `mpstat -P ALL 1` + `pidstat …` during `stream_pipeline --p0 --threads 6 --fanout --windowed --prefetch --neon --entropy --power --overlap 2` (warm; 1 warmup; cooled <58 °C; companions `knee_6_pidstat.log`, `knee_6_stream.log`) | 7856483 | e4ea783 (binary unchanged) | ResFP-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
| results/date27/cpu_probe/ResSHyp/knee_20_mpstat.log | `mpstat -P ALL 1` + `pidstat …` during `stream_pipeline --p0 --threads 20 --fanout --windowed --prefetch --neon --entropy --power --overlap 2` (warm; 1 warmup; cooled <58 °C; companions `knee_20_pidstat.log`, `knee_20_stream.log`) | 7856483 | e4ea783 (binary unchanged) | ResSHyp-relu_s0_L20_pt | 2026-09-01 | no — board binary unchanged since e4ea783 |
