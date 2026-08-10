#!/usr/bin/env python3
"""Cross-language check: decode a C++-written .ddc with the Python oracle (src/utils/ddc_format.py)
and assert the known synthetic fixture from tests/test_ddc_io.cpp. Proves the C++ writer emits bytes
the Python reader understands identically.

Usage:  python inference_cpp/tests/ddc_cross_check.py <path-to.ddc>
"""

import sys
from pathlib import Path

# Import the pure-stdlib oracle without needing the full package on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "utils"))
import ddc_format as ddc

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_ddc_io.ddc"
header, records = ddc.read_ddc(path)

# Must match test_ddc_io.cpp::make_header / make_records exactly.
expected = [(bytes([0xA0 + i]) * i, bytes([0xB0 + i]) * (i + 1)) for i in range(6)]

ok = True


def check(name, cond):
    """Quick general check."""
    global ok
    if not cond:
        print(f"  FAIL: {name}")
        ok = False


check("arch_id", header.arch_id == 3)
check("N/M", header.N == 128 and header.M == 256)
check("patch/stride", header.patch == 256 and header.stride == 256)
check("scene", header.scene_H == 512 and header.scene_W == 768)
check("grid", header.grid_r == 3 and header.grid_a == 2)
check(
    "amp",
    abs(header.amp_min - 4.605170249938965) < 1e-5
    and abs(header.amp_max - 10.742239952087402) < 1e-5
    and abs(header.eps - 0.01) < 1e-6,
)
check("params_sha", header.params_sha == bytes(range(8)))
check("tile_id", header.tile_id == "TEST_TILE")
check("model_id", header.model_id == "ResSHyp-relu_s0_L1000_pt")
check("records", records == expected)

# Exercise the trailer: located at filesize - n*8, random-access the last record.
offsets, table_start = ddc.read_offset_table(path, header)
check("offset table length", len(offsets) == 6)
check("random-access rec5", ddc.read_patch(path, offsets[5]) == expected[5])

print("ddc_cross_check: PASS" if ok else "ddc_cross_check: FAIL")
sys.exit(0 if ok else 1)
