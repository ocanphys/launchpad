"""The guide the web image builds: it copies exactly the modules the docs pages
render with `:::`, since mkdocstrings finds nothing it was not given."""

import re
from pathlib import Path

import main

DOCS = Path(__file__).parent.parent / "docs"


def test_the_guide_build_copies_exactly_the_modules_the_pages_render():
    rendered = {
        target.split(".")[0]
        for page in DOCS.glob("*.md")
        for target in re.findall(r"^::: ([\w.]+)", page.read_text(), re.MULTILINE)
    }
    assert rendered == {source.removesuffix(".py") for source in main.DOCS_SOURCE}
