"""How a call is sized when it is spawned: an artifact's `Resources` as Modal `with_options` kwargs.

Free of FastAPI, so the lab and the workers import it as the leasebook does.
"""

from artifacts.core.artifact import Resources


def resource_options(resources: Resources) -> dict:
    """An artifact's `allocated_resources` -> `Function.with_options()` kwargs.

        resource_options(Resources()) == {}
        resource_options(Resources(cpu=2)) == {"cpu": 2}
        resource_options(Resources(gpu_type="A100")) == {"gpu": "A100"}
        resource_options(Resources(gpu_type="A100", gpu_count=2)) == {"gpu": "A100:2"}

    Unvalidated by design -- `Resources` is only ever constructed by trusted
    code (a notebook declaring an artifact), and Modal itself rejects a bad
    cpu/gpu value server-side at spawn time. Three interpretations worth
    being explicit about instead of leaving to accident:

    - `cpu=None` counts as not declared, same as the field being absent
      entirely. Calling `with_options()` at all, even with every argument at
      its own default, still moves the call into its own dynamically
      configured container pool (see `attempt_launch`), and a no-op override
      is exactly the case not worth paying that for.
    - `gpu_count` with no `gpu_type` is dropped -- nothing to attach a
      count to, so the job runs GPU-less rather than erroring.
    - `gpu_count: 0` reads the same as omitted -- Modal's own "TYPE:COUNT"
      string has no way to ask for zero of a GPU.
    """
    options = {}
    if resources.cpu is not None:
        options["cpu"] = resources.cpu
    if resources.gpu_type:
        options["gpu"] = (
            f"{resources.gpu_type}:{resources.gpu_count}"
            if resources.gpu_count
            else resources.gpu_type
        )
    return options
