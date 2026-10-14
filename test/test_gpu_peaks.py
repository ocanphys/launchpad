"""Consistency checks on the GPU peak table."""
import math

import pytest
from gpus.peaks import gpu_specs, peak_tflops, spec_row


def test_rows_are_unique_and_each_precision_is_positive_or_missing():
    assert gpu_specs.index.is_unique
    assert gpu_specs.columns.tolist() == ["fp64", "fp32", "tf32", "bf16", "fp16", "fp8", "mem_bw"]
    assert ((gpu_specs > 0) | gpu_specs.isna()).all().all()


def test_bf16_equals_fp16_wherever_both_exist_and_fp8_doubles_them():
    both = gpu_specs.dropna(subset=["bf16", "fp16"])
    assert (both["bf16"] == both["fp16"]).all()
    fp8 = gpu_specs.dropna(subset=["fp8", "bf16"])
    assert all(math.isclose(row.fp8, 2 * row.bf16, rel_tol=0.02) for row in fp8.itertuples())


def test_aliases_share_their_compute():
    compute = ["fp64", "fp32", "tf32", "bf16", "fp16", "fp8"]
    assert gpu_specs.loc["A100", compute].equals(gpu_specs.loc["A100-40GB", compute])
    assert gpu_specs.loc["H100!", compute].equals(gpu_specs.loc["H100", compute])
    assert gpu_specs.loc["H200", compute].equals(gpu_specs.loc["H100", compute])


def test_device_names_map_to_the_row_of_the_gpu_modal_gave():
    rows = {
        "Tesla T4": "T4", "NVIDIA L4": "L4", "NVIDIA L40S": "L40S", "NVIDIA A10": "A10",
        "NVIDIA A100-SXM4-40GB": "A100-40GB", "NVIDIA A100-SXM4-80GB": "A100-80GB", "NVIDIA A100 80GB PCIe": "A100-80GB",
        "NVIDIA H100 80GB HBM3": "H100", "NVIDIA H200": "H200", "NVIDIA B200": "B200", "NVIDIA B300 SXM6 AC": "B300",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX-PRO-6000",
    }
    assert {name: spec_row(name) for name in rows} == rows
    assert set(rows.values()) <= set(gpu_specs.index)
    with pytest.raises(KeyError, match="not in gpus/peaks.py"):
        spec_row("NVIDIA A10G")  # a different GPU from the A10: not guessed


def test_peak_tflops_reads_the_precision_column():
    assert peak_tflops("NVIDIA L4", "torch.bfloat16") == 121
    assert peak_tflops("NVIDIA L4", "torch.float32") == 30.3
    assert math.isnan(peak_tflops("Tesla T4", "torch.bfloat16"))  # no bf16 unit
