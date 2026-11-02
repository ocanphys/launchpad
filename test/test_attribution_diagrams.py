"""The pictures on docs/profiling/attribution.md are what attribution_diagrams draws from today's attribution."""
import pytest

from profiling.attribution_diagrams import CASES, OUT, legend, render

REDRAW = "redraw them: uv run python -m profiling.attribution_diagrams"


@pytest.mark.parametrize("name", list(CASES))
def test_each_picture_on_the_docs_page_is_current(name):
    assert (OUT / f"{name}.svg").read_text() == render(CASES[name], gpu_work_only=False), REDRAW


def test_the_legend_is_current():
    assert (OUT / "legend.svg").read_text() == legend(), REDRAW


def test_one_event_is_drawn_with_what_it_launched_under_its_launch_chain():
    picture = render(CASES["depth"], event_id=3)  # aten::bmm, beside aten::add in model.tblocks.0
    assert "forward::model.tblocks.0.ffn" in picture and "aten::bmm" in picture and "gemm" in picture
    assert "aten::add" not in picture


def test_gpu_work_only_leaves_out_what_launches_no_gpu_work():
    rows = CASES["ambiguous-launch"]  # the kernel links to the first aten::mm, and neither launch call queues it
    picture = render(rows)
    assert picture.count("aten::mm") == 1 and "gemm" in picture and "cudaLaunchKernel" not in picture
    assert render(rows, gpu_work_only=False).count("aten::mm") == 2
