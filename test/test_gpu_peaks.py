"""Consistency checks on the GPU peak and price tables."""
import math

import pytest

from gpus.peaks import COLUMNS, gpu_specs, peak_tflops, spec_row
from gpus.pricing import gpu_price_per_s


def test_every_gpu_with_a_peak_has_a_price():
    assert gpu_price_per_s.keys() == gpu_specs.keys()


def test_every_row_has_each_column_positive_or_missing():
    for row in gpu_specs.values():
        assert tuple(row) == COLUMNS
        assert all(value > 0 or math.isnan(value) for value in row.values())


def test_bf16_equals_fp16_wherever_both_exist_and_fp8_doubles_them():
    for row in gpu_specs.values():
        if not math.isnan(row["bf16"]) and not math.isnan(row["fp16"]):
            assert row["bf16"] == row["fp16"]
        if not math.isnan(row["fp8"]) and not math.isnan(row["bf16"]):
            assert math.isclose(row["fp8"], 2 * row["bf16"], rel_tol=0.02)


def test_aliases_share_their_compute():
    compute = [column for column in COLUMNS if column != "mem_bw"]

    def same(a, b):
        return all(gpu_specs[a][c] == gpu_specs[b][c] or math.isnan(gpu_specs[a][c]) and math.isnan(gpu_specs[b][c])
                   for c in compute)

    assert same("A100", "A100-40GB")
    assert same("H100!", "H100")
    assert same("H200", "H100")


def test_device_names_map_to_the_row_of_the_gpu_modal_gave():
    rows = {
        "Tesla T4": "T4", "NVIDIA L4": "L4", "NVIDIA L40S": "L40S", "NVIDIA A10": "A10", "NVIDIA A10G": "A10G",
        "NVIDIA A100-SXM4-40GB": "A100-40GB", "NVIDIA A100-SXM4-80GB": "A100-80GB", "NVIDIA A100 80GB PCIe": "A100-80GB",
        "NVIDIA H100 80GB HBM3": "H100", "NVIDIA H200": "H200", "NVIDIA B200": "B200", "NVIDIA B300 SXM6 AC": "B300",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX-PRO-6000",
    }
    assert {name: spec_row(name) for name in rows} == rows
    assert set(rows.values()) <= set(gpu_specs)
    with pytest.raises(KeyError, match="not in gpus/peaks.py"):
        spec_row("NVIDIA A40")  # a GPU not in the table: not guessed


def test_peak_tflops_reads_the_precision_column():
    assert peak_tflops("NVIDIA L4", "torch.bfloat16") == 121
    assert peak_tflops("NVIDIA L4", "torch.float32") == 30.3
    assert math.isnan(peak_tflops("Tesla T4", "torch.bfloat16"))  # no bf16 unit
