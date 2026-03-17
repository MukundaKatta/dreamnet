"""CLI interface for DreamNet."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from dreamnet.analyzer.comparator import DreamHallucinationComparator
from dreamnet.models import LLMOutput
from dreamnet.report import render_report

console = Console()


@click.group()
def cli() -> None:
    """DreamNet: Explore parallels between AI hallucinations and human dreams."""


@cli.command()
@click.argument("text")
@click.option("--prompt", "-p", default="", help="Original prompt that generated the text.")
@click.option("--ground-truth", "-g", default=None, help="Ground truth for comparison.")
def analyze(text: str, prompt: str, ground_truth: str | None) -> None:
    """Analyze a single LLM output for dream-hallucination parallels."""
    output = LLMOutput(text=text, prompt=prompt, ground_truth=ground_truth)
    comparator = DreamHallucinationComparator()
    result = comparator.analyze(output)
    render_report([result], console)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
def analyze_file(filepath: Path) -> None:
    """Analyze LLM outputs from a JSON file (list of objects with 'text' field)."""
    data = json.loads(filepath.read_text())
    outputs = [LLMOutput(**item) for item in data]
    comparator = DreamHallucinationComparator()
    results = comparator.analyze_batch(outputs)
    render_report(results, console)


if __name__ == "__main__":
    cli()
