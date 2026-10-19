"""Live benchmark display; never performs inference or changes scoring."""

import time


def money(value):
    return "unknown" if value is None else f"${value:.4f}"


def aggregate(rows):
    costs = [row.get("estimated_cost_usd") for row in rows]
    return {
        "completed": len(rows),
        "passed": sum(row["grade"]["passed"] for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "accuracy": sum(row["grade"]["passed"] for row in rows) / len(rows) if rows else None,
        "field_score": sum(row["grade"]["score"] for row in rows) / len(rows) if rows else None,
        "known_cost_usd": sum(value for value in costs if value is not None),
        "unknown_cost_requests": sum(value is None for value in costs),
        "mean_latency_seconds": sum(row["seconds"] for row in rows) / len(rows) if rows else None,
    }


def render(state):
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    rows, total = state["rows"], state["total"]
    stats = aggregate(rows)
    elapsed = max(0.001, time.monotonic() - state["started"])
    eta = elapsed / len(rows) * (total - len(rows)) if rows else None
    known = money(stats["known_cost_usd"])
    if stats["unknown_cost_requests"]:
        known += f" + {stats['unknown_cost_requests']} unknown"
    table = Table(expand=True)
    for title in (
        "Model / harness",
        "Done",
        "Pass",
        "Field score",
        "Errors",
        "Mean s",
        "Est. USD",
        "Active",
    ):
        table.add_column(title, overflow="fold")
    for model in state["models"]:
        name = model["name"]
        subset = [row for row in rows if row["model_name"] == name]
        summary = aggregate(subset)
        cost = money(summary["known_cost_usd"]) if subset else "—"
        if summary["unknown_cost_requests"]:
            cost += " + ?"
        accuracy = summary["accuracy"]
        field = summary["field_score"]
        latency = summary["mean_latency_seconds"]
        active = [value["label"] for value in state["active"].values() if value["model"] == name]
        table.add_row(
            Text(f"{name} ({model['backend']})"),
            f"{len(subset)}/{state['per_model']}",
            "—" if accuracy is None else f"{accuracy:.1%}",
            "—" if field is None else f"{field:.1%}",
            str(summary["errors"]),
            "—" if latency is None else f"{latency:.1f}",
            cost,
            Text(", ".join(active) or "—"),
        )
    recent = Table.grid(expand=True)
    recent.add_column()
    for row in rows[-5:]:
        status = "ERROR" if row.get("error") else "PASS" if row["grade"]["passed"] else "FAIL"
        style = "red" if status != "PASS" else "green"
        recent.add_row(
            Text(
                f"{status:5} {row['model_name']} · {row['case_id']} · "
                f"{row['seconds']:.1f}s · {row.get('error') or row['grade'].get('reason', '')}",
                style=style,
            )
        )
    subtitle = (
        f"{state['status']}  |  {len(rows)}/{total} completed  |  "
        f"elapsed {elapsed:.0f}s  |  ETA {'—' if eta is None else f'{eta:.0f}s'}\n"
        f"Estimated cost so far: {known}  |  "
        f"planned total: {money(state['projected_cost'])}"
    )
    title = "LOCAL VISION BENCHMARK" + (" — SIMULATED DEMO" if state.get("demo") else "")
    return Panel(
        Group(
            Text(subtitle),
            ProgressBar(total=total, completed=len(rows)),
            table,
            Text("Recent results", style="bold"),
            recent,
            Text(
                "Correctness: strict case pass + mean field score; errors count as failures.\n"
                "Costs are estimates, not invoices. Local compute costs excluded.\n"
                "Ctrl-C stops dispatch, cancels CLI calls and drains active local inference.",
                style="dim",
            ),
        ),
        title=title,
    )


class Dashboard:
    def __init__(self, state, mode="auto"):
        self.state, self.mode, self.live, self.console = state, mode, None, None
        self.last_refresh = 0

    def __enter__(self):
        import sys

        enabled = self.mode == "on" or (self.mode == "auto" and sys.stdout.isatty())
        if enabled:
            try:
                from rich.console import Console
                from rich.live import Live
            except ImportError as exc:
                raise RuntimeError("The TUI needs the bench extra: uv sync --extra bench") from exc
            self.console = Console()
            self.live = Live(
                render(self.state),
                console=self.console,
                refresh_per_second=4,
                screen=True,
                auto_refresh=False,
            )
            self.live.start()
        return self

    def update(self, row=None, force=False):
        if self.live:
            if force or time.monotonic() - self.last_refresh >= 0.25:
                self.live.update(render(self.state), refresh=True)
                self.last_refresh = time.monotonic()
        elif row:
            outcome = "ERROR" if row.get("error") else "PASS" if row["grade"]["passed"] else "FAIL"
            print(
                f"[{len(self.state['rows'])}/{self.state['total']}] {row['model_name']} "
                f"{row['case_id']}: {outcome} {row['seconds']:.2f}s "
                f"cost={money(row.get('estimated_cost_usd'))}",
                flush=True,
            )

    def __exit__(self, *_args):
        if self.live:
            self.update(force=True)
            self.live.stop()
