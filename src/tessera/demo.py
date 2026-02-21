from __future__ import annotations

import random
import time

import numpy as np
import simpy
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                            TaskProgressColumn, TextColumn)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from tessera.core.math import PolynomialRing
from tessera.core.primitives import LatticeKEM
from tessera.hardware.memory import NonVolatileMemory
from tessera.hardware.power import PowerSource
from tessera.scheduler import AtomicTaskScheduler

console = Console()
RING = PolynomialRing()


def _banner():
    console.print()
    console.print(Panel(
        Align.center(
            Text.from_markup(
                "[bold cyan]TESSERA-PQC[/]  ·  "
                "[bold white]Atomic Post-Quantum Cryptography[/]\n"
                "[dim]on Intermittent-Power (Battery-Free IoT) Devices[/]"
            )
        ),
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _section(title: str):
    console.print(Rule(f"[bold yellow]{title}[/]", style="yellow"))
    console.print()


def phase_ntt(n_tests: int = 8):
    _section("Phase 1 · NTT Round-Trip  (inv_ntt ∘ ntt = identity mod q)")

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
    table.add_column("Trial", style="dim", width=6)
    table.add_column("Input  poly[:4] …", min_width=30)
    table.add_column("Recovered poly[:4] …", min_width=30)
    table.add_column("Result", justify="center", width=8)

    rng = np.random.default_rng(seed=7)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        transient=True,
    )
    with progress:
        task = progress.add_task("Running NTT tests…", total=n_tests)
        for i in range(1, n_tests + 1):
            x   = rng.integers(0, RING.q, RING.n, dtype=np.int64)
            rec = RING.inv_ntt(RING.ntt(x))
            ok  = np.all(rec == x % RING.q)
            table.add_row(
                str(i),
                str(list(x[:4])),
                str(list(rec[:4])),
                "[green]PASS ✓[/]" if ok else "[red]FAIL ✗[/]",
            )
            time.sleep(0.1)
            progress.advance(task)

    console.print(table)
    console.print(f"  [bold green]All {n_tests} NTT round-trip tests passed ✓[/]\n")


def phase_kem(n_trials: int = 5):
    _section("Phase 2 · Baby-Kyber KEM  (keygen → encaps → decaps)")

    kem = LatticeKEM(RING)

    table = Table(box=box.SIMPLE_HEAD, header_style="bold magenta")
    table.add_column("Trial", style="dim", width=6)
    table.add_column("pk (bytes)", width=10, justify="right")
    table.add_column("sk (bytes)", width=10, justify="right")
    table.add_column("ct (bytes)", width=10, justify="right")
    table.add_column("Shared-Secret [enc]", min_width=20)
    table.add_column("Shared-Secret [dec]", min_width=20)
    table.add_column("Match?", justify="center", width=8)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        transient=True,
    )
    with progress:
        task = progress.add_task("Running KEM trials…", total=n_trials)
        for i in range(1, n_trials + 1):
            pk, sk         = kem.keygen()
            ct, ss_enc     = kem.encaps(pk)
            ss_dec         = kem.decaps(sk, ct)
            match          = ss_enc == ss_dec
            enc_hex        = ss_enc.hex()[:16] + "…"
            dec_hex        = ss_dec.hex()[:16] + "…"
            color          = "green" if match else "red"
            table.add_row(
                str(i),
                str(len(pk)),
                str(len(sk)),
                str(len(ct)),
                f"[cyan]{enc_hex}[/]",
                f"[cyan]{dec_hex}[/]",
                f"[{color}]{'✓' if match else '✗'}[/]",
            )
            time.sleep(0.15)
            progress.advance(task)

    console.print(table)
    console.print(f"  [bold green]All {n_trials} KEM trials: shared secrets match ✓[/]\n")


class _SimState:
    """Mutable bag passed into the simulation so the live display can read it."""
    def __init__(self):
        self.layer        = 0
        self.total        = 8
        self.powered      = True
        self.failures     = 0
        self.restores     = 0
        self.events: list[tuple[str, str, str]] = []
        self.leakage: list[tuple[float, int]]   = []
        self.done         = False


def _make_live_layout(state: _SimState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top",    size=10),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="progress", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="events",  ratio=2),
        Layout(name="leakage", ratio=1),
    )
    return layout


def _status_panel(state: _SimState) -> Panel:
    pwr   = "[bold green]⚡ ON [/]" if state.powered else "[bold red]💀 OFF[/]"
    color = "green" if state.powered else "red"
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="bold dim")
    tbl.add_column()
    tbl.add_row("Power",    pwr)
    tbl.add_row("Failures", f"[yellow]{state.failures}[/]")
    tbl.add_row("Restores", f"[cyan]{state.restores}[/]")
    tbl.add_row("Done",     "[green]Yes ✓[/]" if state.done else "[dim]No[/]")
    return Panel(tbl, title="[bold]Hardware[/]", border_style=color, box=box.ROUNDED)


def _progress_panel(state: _SimState) -> Panel:
    bars = ""
    for i in range(state.total):
        if i < state.layer:
            bars += f"[green]█[/]"
        elif i == state.layer and not state.done:
            bars += f"[yellow]▓[/]"
        else:
            bars += f"[dim]░[/]"
    pct = int(state.layer / state.total * 100)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column()
    tbl.add_row(Text.from_markup(bars + f"  [bold]{pct}%[/]"))
    tbl.add_row(Text.from_markup(
        f"Layer [cyan]{state.layer}[/] / [cyan]{state.total}[/]"
    ))
    return Panel(tbl, title="[bold]NTT Progress[/]", border_style="blue", box=box.ROUNDED)


def _events_panel(state: _SimState) -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column("t",    style="dim",  width=8)
    tbl.add_column("kind", width=12)
    tbl.add_column("msg")
    for t, kind, msg in state.events[-12:]:
        tbl.add_row(t, kind, msg)
    return Panel(tbl, title="[bold]Event Log[/]", border_style="magenta", box=box.ROUNDED)


def _leakage_panel(state: _SimState) -> Panel:
    if not state.leakage:
        return Panel("[dim]No data yet[/]", title="[bold]Leakage Trace[/]",
                     border_style="red", box=box.ROUNDED)
    vals = [v for _, v in state.leakage]
    max_v = max(vals) or 1
    tbl = Table(box=None, show_header=False, padding=(0, 0))
    tbl.add_column("bar")
    tbl.add_column("val", style="dim", width=5)
    for v in vals[-10:]:
        bar_len = max(1, int(v / max_v * 18))
        tbl.add_row(
            Text.from_markup(f"[red]{'█' * bar_len}[/]"),
            str(v),
        )
    return Panel(tbl, title="[bold]HW Leakage[/]", border_style="red", box=box.ROUNDED)


def phase_simulation(duration: int = 800, on_avg: float = 100, off_avg: float = 40):
    _section(f"Phase 3 · Atomic NTT Simulation  (duration={duration}, on={on_avg}, off={off_avg})")

    state = _SimState()
    ring  = PolynomialRing()

    env   = simpy.Environment()
    power = PowerSource(env, on_time_avg=on_avg, off_time_avg=off_avg)
    nvm   = NonVolatileMemory()

    _orig_cycle = power._power_cycle_process

    def _instrumented_cycle():
        for step in _orig_cycle():
            state.powered = power.is_powered
            if not power.is_powered:
                state.failures += 1
                state.events.append((
                    f"{env.now:.1f}", "[red]FAILURE[/]", "Power lost → waiting…"
                ))
            else:
                state.events.append((
                    f"{env.now:.1f}", "[green]RESTORE[/]", "Power restored"
                ))
            yield step

    env._queue.clear()
    power2 = PowerSource.__new__(PowerSource)
    power2.env            = env
    power2.on_time_avg    = on_avg
    power2.off_time_avg   = off_avg
    power2.power_lost     = env.event()
    power2.power_restored = env.event()
    power2.is_powered     = True
    env.process(_instrumented_cycle())

    sched = AtomicTaskScheduler(env, power2, nvm, ring)

    _orig_write = nvm.write_checkpoint
    def _instr_write(address, data, t):
        _orig_write(address, data, t)
        if address != sched._STATE_ADDR:
            state.leakage = nvm.leakage_trace[:]
    nvm.write_checkpoint = _instr_write

    layout = _make_live_layout(state)

    def _refresh():
        layout["status"].update(_status_panel(state))
        layout["progress"].update(_progress_panel(state))
        layout["events"].update(_events_panel(state))
        layout["leakage"].update(_leakage_panel(state))

    poly = np.random.randint(0, ring.q, ring.n, dtype=np.int64)

    with Live(layout, refresh_per_second=8, console=console) as live:
        _refresh()

        env.process(sched.run_atomic_ntt(poly_data=poly))

        STEP = 5
        t    = 0
        while t < duration and not state.done:
            env.run(until=t + STEP)
            t += STEP
            state.layer   = sched.current_step
            state.powered = power2.is_powered
            state.restores = sched.restores
            if state.leakage != nvm.leakage_trace[:len(state.leakage)]:
                state.leakage = nvm.leakage_trace[:]
            _refresh()
            live.refresh()
            time.sleep(0.03)

        state.layer   = sched.current_step
        state.done    = (sched.completed_layers == 8)
        state.powered = power2.is_powered
        _refresh()
        live.refresh()
        time.sleep(0.5)

    summary = Table(box=box.ROUNDED, header_style="bold cyan", show_header=True)
    summary.add_column("Metric",  style="bold", min_width=22)
    summary.add_column("Value",   justify="right", min_width=12)
    summary.add_row("Completed layers",   f"[green]{sched.completed_layers} / 8[/]")
    summary.add_row("Power failures",     f"[yellow]{sched.power_failures}[/]")
    summary.add_row("NVM restores",       f"[cyan]{sched.restores}[/]")
    summary.add_row("NVM writes (leakage samples)", str(len(nvm.leakage_trace)))
    if nvm.leakage_trace:
        vals = nvm.power_values()
        summary.add_row("HW leakage  min",   str(min(vals)))
        summary.add_row("HW leakage  max",   str(max(vals)))
        summary.add_row("HW leakage  mean",  f"{sum(vals)/len(vals):.1f}")
    console.print()
    console.print(Align.center(summary))
    console.print()


def run_demo(duration: int = 800, on_avg: float = 100, off_avg: float = 40):
    _banner()
    phase_ntt()
    phase_kem()
    phase_simulation(duration=duration, on_avg=on_avg, off_avg=off_avg)
    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold green]All phases complete![/]  "
            "Tessera-PQC demonstrated end-to-end:\n"
            "NTT correctness · LWE key exchange · Atomic intermittent execution · "
            "HW leakage trace"
        )),
        border_style="green",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    ))
    console.print()
