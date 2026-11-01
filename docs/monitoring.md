# Monitoring

Every training leg on CUDA reports its MFU and peak memory as step metrics, without the loop
ever waiting on the GPU. For measuring a model on its own, see [Profiling](profiling.md).

- **Window.** `loop_config.gpu_check_every` sets how many steps one window covers.
- **Metrics.** `mfu`, `peak_allocated_pct` and `peak_reserved_pct` (percent of the device's
  VRAM), written to the leg's step log. `mfu` and `peak_alloc` also go to the worker's progress.
- **MFU.** The model's analytic `step_flops()` ([Models](models.md#what-readers-expect-of-a-model))
  over the measured time, against the GPU's dense peak rate for the model's `dtype`, from the
  table in `gpus/peaks.py`. It is not measured if the GPU or that dtype is not in the table.
- **Timing.** A window closes with a CUDA event. `record` polls it on later steps, so a window's
  metrics arrive a few steps after it closes.
- **Off CUDA** nothing is measured.

## Reference

::: gpus.monitor
