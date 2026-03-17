"""Report generation for DreamNet analysis results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dreamnet.models import AnalysisResult


def render_report(results: list[AnalysisResult], console: Console | None = None) -> None:
    """Render a rich report of analysis results to the console."""
    console = console or Console()

    console.print(Panel("[bold cyan]DreamNet Analysis Report[/bold cyan]", expand=False))

    table = Table(title="Hallucination Classification Summary")
    table.add_column("Output (truncated)", style="white", max_width=40)
    table.add_column("Classification", style="magenta")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Dream Mapping", style="yellow")

    for r in results:
        truncated = r.output.text[:60] + ("..." if len(r.output.text) > 60 else "")
        table.add_row(
            truncated,
            r.classification.value,
            f"{r.classification_confidence:.1%}",
            r.dream_mapping.value if r.dream_mapping else "N/A",
        )

    console.print(table)

    # Detail panels
    for i, r in enumerate(results, 1):
        console.print()
        console.print(f"[bold]--- Output {i} ---[/bold]")
        console.print(f"[dim]Reasoning:[/dim] {r.reasoning}")
        if r.parallels:
            console.print(f"[dim]Parallels found:[/dim] {len(r.parallels)}")
            for p in r.parallels:
                console.print(f"  - {p.description[:100]}... (conf: {p.confidence:.0%})")
