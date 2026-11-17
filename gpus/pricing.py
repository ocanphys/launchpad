"""Modal's list price of every GPU it offers: `gpu_price_per_s`, one row per `gpu="..."` string.

    from gpus.pricing import gpu_price_per_s
    gpu_price_per_s["H100"] -> 0.001097
"""

# US dollars per GPU per second, as https://modal.com/pricing lists them (2026-11-12). Rows are the
# strings a GPU is requested with, the same as gpus/peaks.py's gpu_specs. These are base rates: picking
# a region multiplies them by 1.15 to 1.75, non-preemptible execution by 3. An automatic upgrade (A100
# to 80 GB, H100 to H200) keeps the price of what was requested, and B200+ is billed as a B200 on either GPU.
gpu_price_per_s: dict[str, float] = {
    "T4":           0.000164,
    "L4":           0.000222,
    "A10":          0.000306,
    "A10G":         0.000306,  # Modal's earlier name for its A10
    "L40S":         0.000542,
    "A100":         0.000583,
    "A100-40GB":    0.000583,
    "A100-80GB":    0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100":         0.001097,
    "H100!":        0.001097,
    "H200":         0.001261,
    "B200":         0.001736,
    "B200+":        0.001736,
    "B300":         0.001972,
}
