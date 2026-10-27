"""Peak throughput of every GPU Modal offers: `gpu_specs`, one row per `gpu="..."` string.

`peak_tflops` looks one up from what `torch.cuda.get_device_name()` and a torch dtype report.
"""

import re

# Peak dense throughput of every GPU Modal offers, per precision, from the vendor datasheets.
# Rows are the strings a GPU is requested with, gpu_type="...". Compute is in TFLOPs per second (a rate:
# trillions of floating-point operations each second); memory bandwidth in GB/s.
# NaN means the datasheet gives no figure or the GPU has no such unit. Tensor-core rows are dense:
# datasheets quote the 2:4 structured-sparsity rate, twice as high, so those figures were halved.
nan = float("nan")
COLUMNS = ("fp64", "fp32", "tf32", "bf16", "fp16", "fp8", "mem_bw")
gpu_specs: dict[str, dict[str, float]] = {
    row: dict(zip(COLUMNS, values))
    for row, values in {  # fp64   fp32   tf32    bf16    fp16    fp8  mem_bw
        "T4":           (0.25,   8.1,   nan,    nan,     65,   nan,   320),
        "L4":           ( nan,  30.3,    60,    121,    121,   242,   300),
        "A10":          ( nan,  31.2,  62.5,    125,    125,   nan,   600),
        "L40S":         ( nan,  91.6,   183,    362,    362,   733,   864),
        "A100":         (19.5,  19.5,   156,    312,    312,   nan,  1555),  # 40 GB, or an 80 GB upgrade (2039)
        "A100-40GB":    (19.5,  19.5,   156,    312,    312,   nan,  1555),
        "A100-80GB":    (19.5,  19.5,   156,    312,    312,   nan,  2039),
        "RTX-PRO-6000": ( nan,   120,   117,    500,    500,  1000,  1597),  # page figures look sparse; halved
        "H100":         (  67,    67,   494,    989,    989,  1979,  3350),  # SXM; may be upgraded to an H200
        "H100!":        (  67,    67,   494,    989,    989,  1979,  3350),
        "H200":         (  67,    67,   494,    989,    989,  1979,  4800),
        "B200":         (  37,    75,  1125,   2250,   2250,  4500,  8000),
        "B300":         (1.25,    75,  1125,   2250,   2250,  4500,  8000),  # lowest confidence: sources disagree
        "B200+":        ( nan,    75,  1125,   2250,   2250,  4500,  8000),  # a B200 or a B300; fp64 differs
    }.items()
}

# gpu_specs column of each torch dtype, keyed by str(dtype).
PRECISIONS: dict[str, str] = {
    "torch.float64": "fp64", "torch.float32": "fp32", "torch.bfloat16": "bf16", "torch.float16": "fp16",
    "torch.float8_e4m3fn": "fp8", "torch.float8_e5m2": "fp8",
}

# gpu_specs row of each torch.cuda.get_device_name(), first match wins: the A100 pattern with 80GB
# comes before the one without, and \b keeps "L4" from matching "L40S". The Modal-only rows
# (A100, H100!, B200+) are never returned: a device name says which GPU Modal actually gave.
DEVICE_ROWS = {
    "A100-80GB": r"\bA100\b.*80GB", "A100-40GB": r"\bA100\b",
    "T4": r"\bT4\b", "L4": r"\bL4\b", "A10": r"\bA10\b", "L40S": r"\bL40S\b",
    "RTX-PRO-6000": r"RTX PRO 6000", "H100": r"\bH100\b", "H200": r"\bH200\b",
    "B200": r"\bB200\b", "B300": r"\bB300\b",
}


def spec_row(device_name: str) -> str:
    """Return the gpu_specs row for a torch.cuda.get_device_name(); a GPU not in the table raises KeyError."""
    for row, pattern in DEVICE_ROWS.items():
        if re.search(pattern, device_name):
            return row
    raise KeyError(f"{device_name!r} is not in gpus/peaks.py: add a gpu_specs row and a DEVICE_ROWS pattern")


def peak_tflops(device_name: str, dtype: str) -> float:
    """Return the dense peak rate of the GPU for str(dtype) in TFLOPs per second, NaN if it has no unit for it."""
    return gpu_specs[spec_row(device_name)][PRECISIONS[dtype]]
