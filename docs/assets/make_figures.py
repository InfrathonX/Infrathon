#!/usr/bin/env python3
"""Generate the figures in the README.

    docs/assets/overview.svg    how a run works: Work, Controller, Judge and the time budget
    docs/assets/discovery.svg   one 24-hour run on the Quant-VideoGen KV codec task (paper Figure 1)
    docs/assets/results.svg     six agents: best score over the budget on four tasks, rank score over 25
    docs/assets/tasks.svg       the 25 tasks by level and stage (static)

Plain SVG with CSS keyframes only: GitHub plays them in an <img>, needs no script, and the files
follow the viewer's light or dark theme. Everything moves on CSS animations, never SMIL: browsers run
the two on separate clocks that drift apart when the image is scrolled out of view, so a mix of the
two falls out of step. The numbers come from figure_data.json, an
extract of the website's results snapshot (https://infrathonx.github.io). Run this again after
changing the data or the design:

    python3 docs/assets/make_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Roles in the run (overview), agents in the charts (only the top three get a hue; the palette was checked
# for color-vision deficiency on both surfaces), and task stages (task map).
THEME = """
  :root {
    --bg: #ffffff; --card: #f6f8fa; --line: #d0d7de; --text: #1f2328; --muted: #59636e; --faint: #eaeef2;
    --grid: #eef1f4;
    --work: #0969da; --ctrl: #bc4c00; --judge: #0a7f73; --ok: #1a7f37; --bad: #cf222e; --human: #9a6700;
    --work-soft: #ddf4ff; --ctrl-soft: #fff1e5; --judge-soft: #e3f6f3; --bad-soft: #ffebe9;
    --claude: #c24e22; --sol: #2674cb; --astra: #16845e; --other: #9198a1;
    --train: #bc4c00; --train-soft: #fff4ec; --infer: #0969da; --infer-soft: #eef6ff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117; --card: #161b22; --line: #30363d; --text: #e6edf3; --muted: #9198a1; --faint: #21262d;
      --grid: #1b2129;
      --work: #4493f8; --ctrl: #f0883e; --judge: #3fb9a8; --ok: #3fb950; --bad: #f85149; --human: #d29922;
      --work-soft: #132a45; --ctrl-soft: #3a2414; --judge-soft: #12302d; --bad-soft: #3c1618;
      --claude: #d95926; --sol: #3987e5; --astra: #199e70; --other: #656c76;
      --train: #f0883e; --train-soft: #2b1d12; --infer: #4493f8; --infer-soft: #0f2238;
    }
  }
  text { font-family: %s; fill: var(--text); }
  .mono { font-family: %s; }
  .muted { fill: var(--muted); }
  .card { fill: var(--card); stroke: var(--line); stroke-width: 1.2; }
  .line { stroke: var(--line); stroke-width: 1.6; fill: none; }
  .arrow { stroke: var(--muted); stroke-width: 1.6; fill: none; marker-end: url(#head); }
""" % (FONT, MONO)

AGENT_COLOR = {"claude": "--claude", "astra": "--astra", "sol": "--sol"}


def agent_color(key: str) -> str:
    return AGENT_COLOR.get(key, "--other")


class Timeline:
    """CSS keyframes on one shared cycle, written in seconds."""

    def __init__(self, cycle: float, prefix: str) -> None:
        self.cycle = cycle
        self.prefix = prefix
        self.rules: list[str] = []
        self.count = 0

    def pct(self, seconds: float) -> str:
        return f"{max(0.0, min(100.0, 100.0 * seconds / self.cycle)):.3f}%"

    def anim(self, frames: list[tuple[float, str]], extra: str = "") -> str:
        """frames: (seconds, 'prop: value; ...'). Returns a class name to put on an element."""
        self.count += 1
        name = f"{self.prefix}{self.count}"
        frames = sorted(frames, key=lambda f: f[0])
        body = " ".join(f"{self.pct(t)} {{ {css} }}" for t, css in frames)
        self.rules.append(f"@keyframes {name}k {{ {body} }}")
        self.rules.append(f".{name} {{ animation: {name}k {self.cycle}s linear infinite; {extra} }}")
        return name

    def css(self) -> str:
        return "\n  ".join(self.rules)


def window(t: Timeline, start: float, end: float, on: str, off: str, fade: float = 0.25) -> list[tuple[float, str]]:
    """Off, then on between start and end, then off."""
    return [(0, off), (max(0, start - 0.001), off), (start + fade, on), (end, on), (min(t.cycle, end + fade), off),
            (t.cycle, off)]


def windows(t: Timeline, spans: list[tuple[float, float]], on: str, off: str, fade: float = 0.2) -> list[tuple[float, str]]:
    """Like window(), for several non-overlapping spans."""
    frames = [(0.0, off)]
    for start, end in spans:
        frames += [(max(0.0, start - 0.001), off), (start + fade, on), (end, on), (min(t.cycle, end + fade), off)]
    return frames + [(t.cycle, off)]


def hold_from(t: Timeline, start: float, on: str, off: str, fade: float = 0.25) -> list[tuple[float, str]]:
    """Off until start, then on to the end of the cycle (a short reset just before it loops)."""
    return [(0, off), (max(0, start - 0.001), off), (start + fade, on), (t.cycle - 0.35, on), (t.cycle, off)]


def head_marker() -> str:
    return ('<marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill: var(--muted)"/></marker>')


def svg(width: int, height: int, title: str, css: str, body: str, defs: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{title}">\n<title>{title}</title>\n<defs>{head_marker()}{defs}</defs>\n'
            f"<style>{THEME}\n  {css}\n</style>\n"
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" '
            f'style="fill: var(--bg); stroke: var(--line)"/>\n{body}\n</svg>\n')


def header(title: str, sub: str) -> str:
    """Title and a muted subtitle on one line, with room for wider fallback fonts."""
    x = 24 + text_width(title, 19, True) * 1.08 + 14
    return (f'<text x="24" y="40" style="font-size: 19px; font-weight: 650">{title}</text>'
            f'<text x="{x:.0f}" y="40" class="muted" style="font-size: 13.5px">{sub}</text>')


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_width(text: str, size: float, bold: bool = False) -> float:
    """A rough width for the system sans fonts, enough to size pills and avoid collisions."""
    narrow, wide = set("ilj.,:;'|!()[] "), set("mwMW@%")
    units = sum(0.3 if c in narrow else 0.9 if c in wide else 0.7 if c.isupper() or c.isdigit() else 0.58
                for c in text)
    return units * size * (1.12 if bold else 1.0)


def moving_dot(tl: Timeline, points: list[tuple[float, float]], times: list[float], color: str, r: float = 6.5) -> str:
    """A dot that travels the polyline `points`, reaching each vertex at the matching time; hidden otherwise."""
    cycle = tl.cycle
    first, last = times[0], times[-1]

    def at(p: tuple[float, float]) -> str:
        return f"transform: translate({p[0]:.1f}px, {p[1]:.1f}px)"

    move = [(0.0, at(points[0])), (max(0.0, first - 0.002), at(points[0]))]
    move += [(t, at(p)) for t, p in zip(times, points)]
    move += [(min(cycle, last + 0.002), at(points[-1])), (cycle, at(points[-1]))]
    vis = tl.anim([(0, "opacity: 0"), (max(0.0, first - 0.002), "opacity: 0"), (first + 0.06, "opacity: 1"),
                   (last - 0.06, "opacity: 1"), (last, "opacity: 0"), (cycle, "opacity: 0")])
    pos = tl.anim(move)
    return (f'<g class="{vis}"><circle r="{r}" class="{pos}" style="fill: var({color}); stroke: var(--bg); '
            f'stroke-width: 2"/></g>')


def playhead(tl: Timeline, x0: float, x1: float, y0: float, y1: float, t0: float, t1: float, opacity: float) -> str:
    """A vertical line that sweeps from x0 at t0 to x1 at t1, then waits at x1 until the cycle ends."""
    cls = tl.anim([(0, "transform: translateX(0px)"), (t0, "transform: translateX(0px)"),
                   (t1, f"transform: translateX({x1 - x0:.1f}px)"), (tl.cycle, f"transform: translateX({x1 - x0:.1f}px)")])
    return (f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="{cls}" style="stroke: var(--text); '
            f'stroke-width: 1.3; opacity: {opacity}"/>')


def drawn_path(tl: Timeline, points: list[tuple[float, float]], x0: float, x1: float, t0: float, t1: float,
               style: str) -> str:
    """A polyline drawn in step with a playhead that sweeps x0..x1 over t0..t1 (x never decreases along it).
    It stays drawn until just before the cycle loops."""
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1] or 1.0

    def when(x: float) -> float:
        return t0 + (t1 - t0) * (x - x0) / (x1 - x0)

    frames, last_t = [(0.0, total)], 0.0
    for (x, _), run in zip(points, lengths):
        t = max(when(x), last_t + 0.01)          # vertical segments: drawn in 10 ms
        frames.append((t, total - run))
        last_t = t
    frames += [(tl.cycle - 0.4, 0.0), (tl.cycle - 0.05, total), (tl.cycle, total)]
    cls = tl.anim([(t, f"stroke-dashoffset: {off:.1f}") for t, off in frames])
    vis = tl.anim([(0, "opacity: 0"), (max(0.0, when(points[0][0]) - 0.001), "opacity: 0"),
                   (when(points[0][0]) + 0.05, "opacity: 1"), (tl.cycle - 0.05, "opacity: 1"), (tl.cycle, "opacity: 0")])
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)
    return (f'<g class="{vis}"><path d="{d}" class="{cls}" style="{style}; stroke-dasharray: {total + 2:.1f} '
            f'{total + 2:.1f}"/></g>')


# ---- icons (24 x 24, drawn around 0,0 as the top-left) ------------------------------------------

def icon(kind: str, x: float, y: float, color: str, scale: float = 1.0) -> str:
    s = (f'style="fill:none; stroke: var({color}); stroke-width: {1.8 / scale:.2f}; stroke-linecap: round; '
         f'stroke-linejoin: round"')
    shapes = {
        "shield": '<path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
        "gpu": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 6V3M10 6V3M14 6V3M18 6V3M6 21v-3M10 21v-3M14 21v-3M18 21v-3"/>',
        "terminal": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 2.5L7 14M12 14h5"/>',
        "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    }
    return f'<g transform="translate({x},{y}) scale({scale})" {s}>{shapes[kind]}</g>'


def check_mark(cx: float, cy: float, color: str, r: float = 10) -> str:
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" style="fill: var({color})"/>'
            f'<path d="M{cx - r * 0.45} {cy} l{r * 0.33} {r * 0.33} l{r * 0.55} {-r * 0.65}" style="fill: none; '
            f'stroke: var(--bg); stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round"/>')


def cross_mark(cx: float, cy: float, color: str, r: float = 10) -> str:
    d = r * 0.38
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" style="fill: var({color})"/>'
            f'<path d="M{cx - d} {cy - d} L{cx + d} {cy + d} M{cx + d} {cy - d} L{cx - d} {cy + d}" '
            f'style="fill: none; stroke: var(--bg); stroke-width: 2.4; stroke-linecap: round"/>')


# ---- 1. how a run works ----------------------------------------------------------------------------

def overview() -> str:
    W, H, T = 1000, 548, 15.0
    tl = Timeline(T, "o")
    out: list[str] = []
    under: list[str] = []          # drawn below the panels: the moving dots show only in the gaps

    out.append(header("How a run works", "an agent improves a real ML system for 12–24 hours; an independent Judge "
                                         "scores every submission"))

    # The task: tau = (W, P, E, R, T).
    sx0, sy, sh = 112, 58, 40
    seg = (968 - sx0) / 5
    out.append(f'<rect x="24" y="{sy}" width="952" height="{sh}" rx="10" class="card"/>')
    out.append(f'<text x="40" y="{sy + 26}" style="font-size: 15px; font-style: italic; font-family: Georgia, serif">'
               f'task τ =</text>')
    parts = [("W", "Workspace", "--work"), ("P", "Public tests", "--work"), ("E", "Private tests", "--judge"),
             ("R", "Resources", "--muted"), ("T", "Time budget", "--text")]
    for i, (letter, name, color) in enumerate(parts):
        x = sx0 + seg * i
        start = 0.25 + 0.28 * i
        lit = tl.anim(window(tl, start, start + 0.9, "opacity: 1", "opacity: 0", 0.18))
        out.append(f'<rect x="{x + 4:.1f}" y="{sy + 5}" width="{seg - 8:.1f}" height="{sh - 10}" rx="7" class="{lit}" '
                   f'style="fill: var(--faint)"/>')
        if i:
            out.append(f'<path d="M{x:.1f} {sy + 9} V{sy + sh - 9}" class="line" style="stroke-width: 1"/>')
        out.append(f'<text x="{x + 18:.1f}" y="{sy + 26}" style="font-size: 17px; font-style: italic; font-weight: 600; '
                   f'font-family: Georgia, serif; fill: var({color})">{letter}</text>')
        out.append(f'<text x="{x + 46:.1f}" y="{sy + 25}" style="font-size: 13.5px; font-weight: 600">{name}</text>')

    # Panels.
    top, ph = 124, 220
    wx, ww = 24, 352          # Work
    cx, cw = 420, 160         # Controller
    jx, jw = 624, 352         # Judge
    lane = top + ph + 20      # feedback lane
    mid = 220                 # submissions travel at this height

    # The task's public tests go to Work, its private tests to the Judge.
    p_x, e_x = sx0 + seg * 1.5, sx0 + seg * 2.5
    for d in (f"M{p_x:.1f} {sy + sh} V{top - 2}", f"M{e_x:.1f} {sy + sh} V{sy + sh + 11} H{jx + 80} V{top - 2}"):
        out.append(f'<path d="{d}" class="arrow"/>')
        pulse = tl.anim(window(tl, 1.5, 2.3, "opacity: 1; stroke-dashoffset: -40", "opacity: 0; stroke-dashoffset: 0", 0.15))
        out.append(f'<path d="{d}" class="{pulse}" style="fill: none; stroke: var(--text); stroke-width: 2; '
                   f'stroke-dasharray: 6 5"/>')

    def panel(x: float, w: float, color: str, title: str, sub: str, active: list[tuple[float, float]]) -> None:
        halo = tl.anim(windows(tl, active, "opacity: 1", "opacity: 0"))
        out.append(f'<rect x="{x - 4}" y="{top - 4}" width="{w + 8}" height="{ph + 8}" rx="17" class="{halo}" '
                   f'style="fill: none; stroke: var({color}); stroke-width: 2; stroke-opacity: 0.3"/>')
        out.append(f'<rect x="{x}" y="{top}" width="{w}" height="{ph}" rx="14" class="card"/>')
        out.append(f'<rect x="{x}" y="{top}" width="{w}" height="{ph}" rx="14" style="fill: none; stroke: var({color}); '
                   f'stroke-opacity: 0.55; stroke-width: 1.4"/>')
        out.append(f'<circle cx="{x + 18}" cy="{top + 21}" r="5" style="fill: var({color})"/>')
        out.append(f'<text x="{x + 30}" y="{top + 26}" style="font-size: 15px; font-weight: 650">{title}</text>')
        if sub:
            out.append(f'<text x="{x + 34 + text_width(title, 15, True):.0f}" y="{top + 26}" class="muted" '
                       f'style="font-size: 12px">{sub}</text>')

    # Story: three submissions. The first passes, the second breaks a rule, the third passes with a better score.
    edit = [(2.0, 3.5), (7.6, 8.0), (10.5, 10.9)]
    test = [(3.5, 4.3), (8.0, 8.7), (10.9, 11.2)]
    submit = [4.3, 8.7, 11.2]
    review = [(4.8, 5.4), (9.2, 9.8), (11.7, 12.1)]
    judge_on = [(5.9, 7.0), (12.5, 13.4)]
    result = [7.0, 9.8, 13.4]

    panel(wx, ww, "--work", "Work", "the agent's container and GPUs",
          [(2.0, 4.5), (7.4, 8.9), (10.3, 11.4)])
    panel(cx, cw, "--ctrl", "Controller", "", [(4.6, 5.6), (9.0, 10.0), (11.5, 12.3)])
    panel(jx, jw, "--judge", "Judge", "its own container and GPUs", [(5.7, 7.1), (12.3, 13.5)])

    # Work: an editor and the public tests.
    ex, ey, ew, eh = wx + 16, top + 42, 214, 132
    out.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="9" style="fill: var(--bg); stroke: var(--line)"/>')
    out.append(f'<path d="M{ex} {ey + 22} H{ex + ew}" class="line" style="stroke-width: 1"/>')
    for i, c in enumerate(["--bad", "--ctrl", "--ok"]):
        out.append(f'<circle cx="{ex + 13 + 11 * i}" cy="{ey + 11}" r="3.5" style="fill: var({c}); opacity: .7"/>')
    out.append(f'<text x="{ex + 50}" y="{ey + 15.5}" class="muted mono" style="font-size: 11px">agent</text>')
    widths = [150, 108, 170, 90, 134, 62]
    colors = ["--work", "--muted", "--judge", "--muted", "--ctrl", "--muted"]
    for i, width in enumerate(widths):
        begin = 2.05 + 0.2 * i
        frames = [(0, "transform: scaleX(0)"), (begin, "transform: scaleX(0)"), (begin + 0.3, "transform: scaleX(1)")]
        if i in (2, 4):                                     # lines the agent rewrites before later submissions
            for s, _ in edit[1:]:
                b = s + 0.05 + 0.1 * (i == 4)
                frames += [(b, "transform: scaleX(1)"), (b + 0.08, "transform: scaleX(0.1)"), (b + 0.35, "transform: scaleX(1)")]
        frames += [(T - 0.4, "transform: scaleX(1)"), (T - 0.05, "transform: scaleX(0)"), (T, "transform: scaleX(0)")]
        cls = tl.anim(frames, "transform-box: fill-box; transform-origin: left center;")
        out.append(f'<rect x="{ex + 14 + (12 if i in (2, 3) else 0)}" y="{ey + 36 + 15 * i}" width="{width}" height="6" '
                   f'rx="3" class="{cls}" style="fill: var({colors[i]}); opacity: 0.6"/>')

    tx, tw_ = ex + ew + 12, wx + ww - 16 - (ex + ew + 12)
    out.append(f'<rect x="{tx}" y="{ey}" width="{tw_}" height="{eh}" rx="9" style="fill: var(--bg); stroke: var(--line)"/>')
    out.append(f'<text x="{tx + tw_ / 2}" y="{ey + 17}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'public tests</text>')
    for k in range(6):
        col, row = k % 2, k // 2
        bx, by = tx + 14 + col * (tw_ - 28 - 30) + 0, ey + 32 + row * 30
        green_spans = []
        for n, (s, _) in enumerate(test):
            off = edit[n + 1][0] if n + 1 < len(edit) else T - 0.4
            green_spans.append((s + 0.1 + 0.08 * k, off))
        cls = tl.anim(windows(tl, green_spans, "opacity: 1", "opacity: 0", 0.12))
        out.append(f'<rect x="{bx:.1f}" y="{by}" width="30" height="20" rx="4" style="fill: var(--faint)"/>')
        out.append(f'<rect x="{bx:.1f}" y="{by}" width="30" height="20" rx="4" class="{cls}" style="fill: var(--ok); '
                   f'opacity: 0"/>')

    # What the agent is doing.
    status = [("editing the system", edit), ("running the public tests", test),
              ("submitting", [(s, s + 0.5) for s in submit]),
              ("reading the Judge's feedback", [(7.0, 7.6), (10.0, 10.5), (13.6, T - 0.4)])]
    for label, spans in status:
        cls = tl.anim(windows(tl, spans, "opacity: 1", "opacity: 0", 0.1))
        out.append(f'<g class="{cls}"><text x="{ex}" y="{top + ph - 18}" class="mono" style="font-size: 12px; '
                   f'fill: var(--work)">›</text><text x="{ex + 13}" y="{top + ph - 18}" class="muted mono" '
                   f'style="font-size: 12px">{label}</text></g>')

    # Controller: the integrity review.
    rx_, ry, rw, rh = cx + 14, top + 40, cw - 28, 104
    out.append(f'<rect x="{rx_}" y="{ry}" width="{rw}" height="{rh}" rx="9" style="fill: var(--bg); stroke: var(--line)"/>')
    out.append(icon("shield", cx + cw / 2 - 13, ry + 12, "--ctrl", 1.1))
    out.append(f'<text x="{cx + cw / 2}" y="{ry + 64}" text-anchor="middle" style="font-size: 13px; font-weight: 650">'
               f'integrity review</text>')
    out.append(f'<text x="{cx + cw / 2}" y="{ry + 82}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'an LLM reads the code</text>')
    scan_frames: list[tuple[float, str]] = [(0, "opacity: 0; transform: translateY(0)")]
    for s, e in review:
        scan_frames += [(s - 0.001, "opacity: 0; transform: translateY(0)"), (s + 0.05, "opacity: 1; transform: translateY(0)"),
                        (e - 0.05, f"opacity: 1; transform: translateY({rh - 8}px)"),
                        (e, f"opacity: 0; transform: translateY({rh - 8}px)")]
    scan_frames.append((T, "opacity: 0; transform: translateY(0)"))
    scan = tl.anim(scan_frames)
    out.append(f'<rect x="{rx_ + 6}" y="{ry + 4}" width="{rw - 12}" height="2.5" rx="1.2" class="{scan}" '
               f'style="fill: var(--ctrl)"/>')
    ok = tl.anim(windows(tl, [(review[0][1], review[0][1] + 1.0), (review[2][1], review[2][1] + 1.0)], "opacity: 1",
                         "opacity: 0", 0.1))
    bad = tl.anim(window(tl, review[1][1], review[1][1] + 1.2, "opacity: 1", "opacity: 0", 0.1))
    out.append(f'<g class="{ok}">{check_mark(rx_ + rw - 4, ry + 4, "--ok")}</g>')
    out.append(f'<g class="{bad}">{cross_mark(rx_ + rw - 4, ry + 4, "--bad")}</g>')
    calm = tl.anim(window(tl, review[1][1], review[1][1] + 1.2, "opacity: 0", "opacity: 1", 0.1))
    out.append(f'<g class="{calm}"><text x="{cx + cw / 2}" y="{ry + rh + 26}" text-anchor="middle" class="muted" '
               f'style="font-size: 11.5px">against the task\'s rules,</text><text x="{cx + cw / 2}" y="{ry + rh + 42}" '
               f'text-anchor="middle" class="muted" style="font-size: 11.5px">before any GPU time</text></g>')
    out.append(f'<g class="{bad}"><text x="{cx + cw / 2}" y="{ry + rh + 26}" text-anchor="middle" '
               f'style="font-size: 11.5px; font-weight: 600; fill: var(--bad)">rejected: caches</text>'
               f'<text x="{cx + cw / 2}" y="{ry + rh + 42}" text-anchor="middle" style="font-size: 11.5px; '
               f'font-weight: 600; fill: var(--bad)">results across calls</text></g>')

    # Judge: held-out workloads, then execute, validate, score.
    out.append(f'<text x="{jx + 16}" y="{top + 56}" class="muted" style="font-size: 11.5px">private tests</text>')
    n_cells = 6
    for k in range(n_cells):
        x = jx + 100 + 30 * k
        spans = [(s + 0.12 * k, e) for s, e in judge_on]
        cls = tl.anim(windows(tl, spans, "opacity: 1", "opacity: 0.22", 0.1))
        out.append(f'<rect x="{x}" y="{top + 42}" width="24" height="20" rx="4" class="{cls}" style="fill: var(--judge)"/>')
    out.append(f'<text x="{jx + 100 + 30 * n_cells + 4}" y="{top + 57}" style="font-size: 13px; font-style: italic; '
               f'font-family: Georgia, serif" class="muted">x₁ … xₙ</text>')
    steps = ["execute", "validate", "score"]
    chip_w, gap = 92, 24
    for k, name in enumerate(steps):
        x = jx + 16 + k * (chip_w + gap)
        y = top + 82
        spans = [(s + 0.35 * k + 0.1, s + 0.35 * k + 0.6) for s, _ in judge_on]
        cls = tl.anim(windows(tl, spans, "opacity: 1", "opacity: 0", 0.1))
        out.append(f'<rect x="{x}" y="{y}" width="{chip_w}" height="28" rx="8" style="fill: var(--bg); stroke: var(--line)"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{chip_w}" height="28" rx="8" class="{cls}" style="fill: var(--judge-soft); '
                   f'stroke: var(--judge); stroke-width: 1.5"/>')
        out.append(f'<text x="{x + chip_w / 2}" y="{y + 18.5}" text-anchor="middle" style="font-size: 12.5px; '
                   f'font-weight: 600">{name}</text>')
        if k:
            out.append(f'<path d="M{x - gap + 3} {y + 14} H{x - 3}" class="arrow"/>')
    out.append(f'<text x="{jx + 16}" y="{top + 136}" class="muted" style="font-size: 11.5px">checks correctness and '
               f'fidelity, then measures</text>')
    # The latest result: a meter that fills to the submission's score.
    my = top + 160
    out.append(f'<text x="{jx + 16}" y="{my + 10}" style="font-size: 12.5px; font-weight: 600">score</text>')
    mx0, mw = jx + 62, jw - 62 - 52
    out.append(f'<rect x="{mx0}" y="{my}" width="{mw}" height="12" rx="6" style="fill: var(--faint)"/>')
    meter = tl.anim([(0, "transform: scaleX(0)"), (result[0] - 0.4, "transform: scaleX(0)"),
                     (result[0], "transform: scaleX(0.42)"), (result[2] - 0.4, "transform: scaleX(0.42)"),
                     (result[2], "transform: scaleX(0.9)"), (T - 0.35, "transform: scaleX(0.9)"),
                     (T, "transform: scaleX(0)")], "transform-box: fill-box; transform-origin: left center;")
    out.append(f'<rect x="{mx0}" y="{my}" width="{mw}" height="12" rx="6" class="{meter}" style="fill: var(--judge)"/>')
    valid = tl.anim(windows(tl, [(result[0] - 0.3, 9.4), (result[2] - 0.3, T - 0.4)], "opacity: 1", "opacity: 0", 0.1))
    out.append(f'<g class="{valid}">{check_mark(jx + jw - 26, my + 6, "--ok", 9)}</g>')
    out.append(f'<text x="{jx + 16}" y="{top + ph - 14}" class="muted" style="font-size: 11.5px">held-out workloads '
               f'the agent never sees</text>')

    # Arrows between the panels, and the feedback lane under them.
    out.append(f'<path d="M{wx + ww + 2} {mid} H{cx - 3}" class="arrow"/>')
    out.append(f'<text x="{(wx + ww + cx) / 2}" y="{mid - 8}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'submit</text>')
    out.append(f'<path d="M{cx + cw + 2} {mid} H{jx - 3}" class="arrow"/>')
    out.append(f'<text x="{(cx + cw + jx) / 2}" y="{mid - 8}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'pass</text>')
    fb_x = wx + ww * 0.55
    out.append(f'<path d="M{jx + jw * 0.5} {top + ph + 2} V{lane} H{fb_x} V{top + ph + 4}" class="arrow" '
               f'style="stroke-dasharray: 4 4"/>')
    out.append(f'<path d="M{cx + cw / 2} {top + ph + 2} V{lane}" class="line" style="stroke-dasharray: 4 4; '
               f'stroke: var(--muted)"/>')
    out.append(f'<text x="{cx + cw / 2}" y="{lane + 17}" text-anchor="middle" class="muted" '
               f'style="font-size: 11.5px">feedback to the agent: the score, or why the submission was rejected</text>')

    # Dots: a submission to the Controller and the Judge, and the feedback back to Work.
    for n in (0, 2):
        s = submit[n]
        r0, r1 = review[n]
        j0 = judge_on[0 if n == 0 else 1][0]
        under.append(moving_dot(tl, [(wx + ww - 40, mid), (cx + cw / 2, mid), (cx + cw / 2, mid), (jx + 60, mid)],
                                [s, r0, r1, j0], "--work"))
        back = result[n]
        under.append(moving_dot(tl, [(jx + jw * 0.5, top + ph - 10), (jx + jw * 0.5, lane), (fb_x, lane),
                                     (fb_x, top + ph - 10)], [back, back + 0.15, back + 0.5, back + 0.6], "--judge"))
    s = submit[1]
    r0, r1 = review[1]
    under.append(moving_dot(tl, [(wx + ww - 40, mid), (cx + cw / 2, mid)], [s, r0], "--work"))
    under.append(moving_dot(tl, [(cx + cw / 2, top + ph - 10), (cx + cw / 2, lane), (fb_x, lane), (fb_x, top + ph - 10)],
                            [r1, r1 + 0.12, r1 + 0.45, r1 + 0.55], "--bad"))

    # The time budget: a playhead, each result as it comes back, and the best valid score so far.
    by0, by1 = 414, 518                       # band top, axis
    ax0, ax1 = 236, 952
    p0, p1 = 2.0, 14.0

    def at(t: float) -> float:
        return ax0 + (ax1 - ax0) * max(0.0, min(1.0, (t - p0) / (p1 - p0)))

    out.append(icon("clock", 24, by0 + 26, "--muted", 0.9))
    out.append(f'<text x="50" y="{by0 + 42}" style="font-size: 13.5px; font-weight: 650">time budget</text>')
    out.append(f'<text x="50" y="{by0 + 60}" class="muted" style="font-size: 12px">12 or 24 hours</text>')
    out.append(f'<text x="50" y="{by0 + 78}" class="muted" style="font-size: 12px">best valid score so far</text>')
    for q in range(5):
        x = ax0 + (ax1 - ax0) * q / 4
        out.append(f'<path d="M{x:.1f} {by1} V{by1 + 5}" class="line"/>')
    out.append(f'<text x="{ax0}" y="{by1 + 20}" text-anchor="middle" class="muted" style="font-size: 11.5px">0</text>')
    out.append(f'<text x="{ax1}" y="{by1 + 20}" text-anchor="middle" class="muted" style="font-size: 11.5px">T</text>')
    # The two valid results, at the heights of the Judge's score meter (0.42 and 0.9 of the band).
    band = by1 - by0 - 12
    lv1, lv3 = by1 - 0.42 * band, by1 - 0.9 * band
    step = [(ax0, by1), (at(result[0]), by1), (at(result[0]), lv1), (at(result[2]), lv1), (at(result[2]), lv3), (ax1, lv3)]
    # The shaded area under the step, shown in pieces as the playhead reaches each result.
    for (xa, ya), xb, t in ((step[2], at(result[2]), result[0]), (step[4], ax1, result[2])):
        cls = tl.anim([(0, "opacity: 0; transform: scaleX(0)"), (t - 0.001, "opacity: 0; transform: scaleX(0)"),
                       (t + 0.05, "opacity: 1; transform: scaleX(0)"),
                       (t + (p1 - p0) * (xb - xa) / (ax1 - ax0), "opacity: 1; transform: scaleX(1)"),
                       (T - 0.35, "opacity: 1; transform: scaleX(1)"), (T, "opacity: 0; transform: scaleX(1)")],
                      "transform-box: fill-box; transform-origin: left center;")
        out.append(f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa:.1f}" height="{by1 - ya:.1f}" class="{cls}" '
                   f'style="fill: var(--judge); opacity: 0; fill-opacity: 0.12"/>')
    out.append(drawn_path(tl, step, ax0, ax1, p0, p1, "fill: none; stroke: var(--judge); stroke-width: 2.4; "
                                                    "stroke-linejoin: round"))
    out.append(f'<path d="M{ax0} {by1} H{ax1}" class="line"/>')
    for t, y, good in ((result[0], lv1, True), (result[1], by1 - 10, False), (result[2], lv3, True)):
        cls = tl.anim(hold_from(tl, t, "opacity: 1; transform: scale(1)", "opacity: 0; transform: scale(0.3)", 0.2),
                      "transform-box: fill-box; transform-origin: center;")
        mark = check_mark(at(t), y, "--ok", 8) if good else cross_mark(at(t), y, "--bad", 8)
        out.append(f'<g class="{cls}">{mark}</g>')
    out.append(playhead(tl, ax0, ax1, by0 - 6, by1, p0, p1, 0.55))
    best = tl.anim(hold_from(tl, 13.6, "opacity: 1", "opacity: 0.0", 0.3))
    out.append(f'<g class="{best}"><text x="{ax1 - 14}" y="{lv3 - 12:.1f}" text-anchor="end" style="font-size: 12.5px; '
               f'font-weight: 650; fill: var(--judge)">the run\'s result: its best valid score</text></g>')

    out[0:0] = under
    return svg(W, H, "How an Infrathon run works: the agent submits from Work, the Controller reviews, the Judge scores, "
                     "and the run keeps its best valid score within the time budget", tl.css(), "\n".join(out))



# ---- 2. one 24-hour run on Quant-VideoGen KV codec -----------------------------------------------------

def task(key: str) -> dict:
    return next(t for t in DATA["tasks"] if t["key"] == key)


def model_name(key: str) -> str:
    return next(m["name"] for m in DATA["models"] if m["key"] == key)


def step_points(points: list[list[float]], end_h: float, fx, fy) -> list[tuple[float, float]]:
    """Best score so far as a step line: flat until the next better submission, then up."""
    out = [(fx(points[0][0]), fy(points[0][1]))]
    for (h, v) in points[1:]:
        out += [(fx(h), out[-1][1]), (fx(h), fy(v))]
    return out + [(fx(end_h), out[-1][1])]


def discovery() -> str:
    W, H, T = 1000, 486, 21.0
    play = 15.0                                   # seconds for the 24 hours
    tl = Timeline(T, "d")
    qvg = task("qvg")
    human = 6.82                                  # the human codec's task score
    out: list[str] = []

    def at(hour: float) -> float:
        return play * hour / 24.0

    out.append(header("24 hours on one task", "Quant-VideoGen KV codec: compress the KV cache of two video models "
                                              "without hurting the video"))

    px0, px1, py0, py1, vmax = 72, 704, 420, 92, 30.0

    def fx(h: float) -> float:
        return px0 + (px1 - px0) * h / 24.0

    def fy(v: float) -> float:
        return py0 - (py0 - py1) * v / vmax

    out.append(f'<text x="{px0}" y="{py1 - 14}" class="muted" style="font-size: 12px">score: compression, weighted by '
               f'video quality</text>')
    for v in (0, 10, 20, 30):
        out.append(f'<path d="M{px0} {fy(v):.1f} H{px1}" style="stroke: var(--grid); stroke-width: 1"/>'
                   f'<text x="{px0 - 10}" y="{fy(v) + 4:.1f}" text-anchor="end" class="muted" style="font-size: 12px">{v}</text>')
    for h in range(0, 25, 4):
        out.append(f'<text x="{fx(h):.1f}" y="{py0 + 20}" text-anchor="middle" class="muted" style="font-size: 12px">'
                   f'{h} h</text>')
    out.append(f'<path d="M{px0} {py0} H{px1}" style="stroke: var(--line); stroke-width: 1.2"/>')

    # The human expert's codec.
    out.append(f'<path d="M{px0} {fy(human):.1f} H{px1}" style="stroke: var(--human); stroke-width: 1.6; '
               f'stroke-dasharray: 6 5"/>')
    out.append(f'<text x="{px0 - 10}" y="{fy(human) + 4:.1f}" text-anchor="end" style="font-size: 11.5px; font-weight: 650; '
               f'fill: var(--human)">human</text>')

    # Every agent's best score so far, drawn as the playhead passes.
    order = ["glm", "deepseek", "qwen", "sol", "astra", "claude"]
    for key in order:
        pts = qvg["traj"][key]
        width = 2.8 if key == "claude" else 1.9 if key in AGENT_COLOR else 1.5
        out.append(drawn_path(tl, step_points(pts, 24, fx, fy), px0, px1, 0.0, play,
                              f"fill: none; stroke: var({agent_color(key)}); stroke-width: {width}; "
                              f"stroke-linejoin: round; stroke-linecap: round"))

    # Claude's submissions the Judge found invalid, as small crosses on the time axis.
    bad = [h for h, v, status in DATA["qvg_claude"]["submissions"] if status != "valid"]
    for h in bad:
        cls = tl.anim(hold_from(tl, at(h), "opacity: 1", "opacity: 0", 0.15))
        out.append(f'<path class="{cls}" d="M{fx(h) - 3.5:.1f} {py0 - 12.5} l7 7 m0 -7 l-7 7" style="stroke: var(--bad); '
                   f'stroke-width: 1.8; stroke-linecap: round; fill: none"/>')
    first_bad = [h for h in bad if h < 6.5]
    note = tl.anim(hold_from(tl, at(first_bad[-1]) + 0.1, "opacity: 1", "opacity: 0", 0.3))
    out.append(f'<text x="{fx(first_bad[0]) - 4:.1f}" y="{py0 - 21}" class="muted {note}" style="font-size: 11.5px">'
               f'{len(first_bad)} invalid tries first</text>')

    # Claude's milestones (paper Figure 1): what changed, and the compression it reached.
    steps = [(6.86, "4-bit quantization", "3.56×"), (8.18, "PCA + entropy coding", "8.87×"),
             (9.35, "quantization-step tuning", "12.23×"), (17.6, "trellis quantization", "17.17×"),
             (23.81, "metadata compression", "19.62×", "+ per-head quantization")]
    claude = qvg["traj"]["claude"]

    def score_at(hour: float) -> float:
        return max(v for h, v in claude if h <= hour + 1e-6)

    boxes = [(88, 262), (88, 206), (88, 150), (384, 234), (418, 98)]      # top-left of each callout
    for n, (step_, (bx, by)) in enumerate(zip(steps, boxes), start=1):
        hour, what, ratio = step_[0], step_[1], step_[2]
        extra = step_[3] if len(step_) > 3 else ""
        mx, my = fx(hour), fy(score_at(hour))
        bw = max(text_width(what, 12.5, True), text_width(extra, 12.5, True)) + 26
        bh = 44 if not extra else 60
        cls = tl.anim(hold_from(tl, at(hour), "opacity: 1; transform: translateY(0)", "opacity: 0; transform: translateY(4px)",
                                0.3), "transform-box: fill-box;")
        anchor_x = bx + bw if mx > bx + bw else mx
        anchor_y = by + bh / 2 if mx > bx + bw else by if my < by else by + bh
        out.append(f'<g class="{cls}">')
        out.append(f'<path d="M{mx:.1f} {my:.1f} L{anchor_x:.1f} {anchor_y:.1f}" style="stroke: var(--claude); '
                   f'stroke-width: 1.1; stroke-dasharray: 3 3; fill: none"/>')
        out.append(f'<rect x="{bx}" y="{by}" width="{bw:.0f}" height="{bh}" rx="8" style="fill: var(--bg); '
                   f'stroke: var(--line)"/>')
        out.append(f'<text x="{bx + 12}" y="{by + 18}" style="font-size: 12.5px; font-weight: 600">{esc(what)}</text>')
        if extra:
            out.append(f'<text x="{bx + 12}" y="{by + 34}" style="font-size: 12.5px; font-weight: 600">{esc(extra)}</text>')
        out.append(f'<text x="{bx + 12}" y="{by + bh - 10}" class="muted" style="font-size: 12px">'
                   f'<tspan style="font-weight: 700; fill: var(--text)">{ratio}</tspan> compression</text>')
        out.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="9" style="fill: var(--claude); stroke: var(--bg); stroke-width: 2"/>'
                   f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" style="font-size: 11px; font-weight: 700; '
                   f'fill: var(--bg)">{n}</text>')
        out.append('</g>')

    # Playhead and clock.
    out.append(playhead(tl, px0, px1, py1 - 4, py0, 0.0, play, 0.45))
    for h in range(25):
        s0, s1 = at(h), (at(h + 1) if h < 24 else T - 0.4)
        cls = tl.anim([(0, "opacity: 0"), (max(0, s0 - 0.001), "opacity: 0"), (s0 + 0.001, "opacity: 1"),
                       (s1, "opacity: 1"), (min(T, s1 + 0.001), "opacity: 0"), (T, "opacity: 0")])
        out.append(f'<text x="{px1}" y="{py1 - 14}" text-anchor="end" class="{cls}" style="font-size: 12.5px">'
                   f'<tspan style="font-weight: 700">{h} h</tspan><tspan class="muted"> of 24</tspan></text>')

    # Scoreboard: every agent's best score so far, bars that follow the playhead.
    sx, sw = 752, 216
    out.append(f'<text x="{sx}" y="{py1 - 14}" style="font-size: 13px; font-weight: 650">best score so far</text>')
    final = sorted(qvg["traj"], key=lambda k: -qvg["traj"][k][-1][1])
    row_h = 42
    for i, key in enumerate(final):
        y = py1 + 6 + row_h * i
        pts = qvg["traj"][key]
        out.append(f'<text x="{sx}" y="{y + 12}" style="font-size: 12.5px">{model_name(key)}</text>')
        out.append(f'<rect x="{sx}" y="{y + 20}" width="{sw}" height="10" rx="5" style="fill: var(--faint)"/>')
        frames = [(0.0, "transform: scaleX(0)")]
        prev = 0.0
        for h, v in pts:
            frames += [(at(h) - 0.001, f"transform: scaleX({prev / vmax:.4f})"), (at(h) + 0.12, f"transform: scaleX({v / vmax:.4f})")]
            prev = v
        frames += [(T - 0.4, f"transform: scaleX({prev / vmax:.4f})"), (T, "transform: scaleX(0)")]
        cls = tl.anim(frames, "transform-box: fill-box; transform-origin: left center;")
        out.append(f'<rect x="{sx}" y="{y + 20}" width="{sw}" height="10" rx="5" class="{cls}" '
                   f'style="fill: var({agent_color(key)})"/>')
        val = tl.anim(hold_from(tl, play, "opacity: 1", "opacity: 0", 0.3))
        out.append(f'<text x="{sx + sw}" y="{y + 12}" text-anchor="end" class="{val}" style="font-size: 12.5px; '
                   f'font-weight: 700">{pts[-1][1]:.2f}</text>')
    hx = sx + sw * human / vmax
    for i in range(len(final)):
        y = py1 + 6 + row_h * i
        out.append(f'<path d="M{hx:.1f} {y + 16} V{y + 34}" style="stroke: var(--human); stroke-width: 1.6; '
                   f'stroke-dasharray: 3 2"/>')
    out.append(f'<text x="{hx:.1f}" y="{py1 + 6 + row_h * len(final) + 10}" text-anchor="middle" style="font-size: 11.5px; '
               f'font-weight: 650; fill: var(--human)">human codec {human}</text>')

    # The outcome, once the 24 hours are up.
    done = tl.anim(hold_from(tl, play + 0.3, "opacity: 1; transform: translateY(0)", "opacity: 0; transform: translateY(6px)",
                             0.4), "transform-box: fill-box;")
    oy = py1 + 6 + row_h * len(final) + 30
    out.append(f'<g class="{done}"><rect x="{sx - 8}" y="{oy}" width="{sw + 16}" height="64" rx="10" '
               f'style="fill: var(--card); stroke: var(--claude); stroke-opacity: .6"/>'
               f'<text x="{sx + 4}" y="{oy + 22}" style="font-size: 13px; font-weight: 700">2.9× the human codec\'s</text>'
               f'<text x="{sx + 4}" y="{oy + 38}" style="font-size: 13px; font-weight: 700">compression, with better</text>'
               f'<text x="{sx + 4}" y="{oy + 54}" class="muted" style="font-size: 12px">PSNR, SSIM and LPIPS</text></g>')

    out.append(f'<text x="{px0}" y="{H - 16}" class="muted" style="font-size: 11.5px">One 24-hour run per agent. '
               f'Callouts follow paper Figure 1 (Claude Opus 5.5); × marks its submissions the Judge rejected as invalid.</text>')
    return svg(W, H, "24 hours on the Quant-VideoGen KV codec task: six agents' best scores over time, and the steps by "
                     "which Claude Opus 5.5 overtook the human codec", tl.css(), "\n".join(out))



# ---- 3. six agents, 25 tasks ---------------------------------------------------------------------------

def nice_ticks(lo: float, hi: float, count: int = 3) -> list[float]:
    raw = (hi - lo) / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 6))
        v += step
    return ticks


def results() -> str:
    W, H, T = 1000, 500, 16.0
    play = 10.0
    tl = Timeline(T, "r")
    out: list[str] = []
    out.append(header("Six agents, 25 tasks", "best valid score as the time budget runs out, on four of the tasks; "
                                              "rank score over all 25"))

    # Legend: three agents have a hue, the other three share gray.
    lx = 24
    for key in ("claude", "astra", "sol"):
        out.append(f'<path d="M{lx} 66 h18" style="stroke: var({agent_color(key)}); stroke-width: 3; stroke-linecap: round"/>'
                   f'<text x="{lx + 25}" y="70.5" style="font-size: 12.5px">{model_name(key)}</text>')
        lx += 25 + text_width(model_name(key), 12.5) + 22
    out.append(f'<path d="M{lx} 66 h18" style="stroke: var(--other); stroke-width: 2; stroke-linecap: round"/>'
               f'<text x="{lx + 25}" y="70.5" class="muted" style="font-size: 12.5px">Qwen 3.8 Flash, DeepSeek V4.1 Flash, '
               f'GLM 5.3 Flash</text>')

    panels = [("sageattention", "SageAttention", "inference · operator · TFLOP/s"),
              ("orpo", "Qwen3 ORPO", "training · module · score"),
              ("opcd", "Miles Qwen3 OPCD", "training · system · score"),
              ("llada", "LLaDA decoding", "inference · system · tokens/s")]
    pw, ph = 296, 192
    for n, (key, title, sub) in enumerate(panels):
        t = task(key)
        budget = t["budget_h"]
        x, y = 24 + (n % 2) * (pw + 16), 88 + (n // 2) * (ph + 14)
        out.append(f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="12" class="card"/>')
        out.append(f'<text x="{x + 14}" y="{y + 23}" style="font-size: 13.5px; font-weight: 650">{title}</text>')
        out.append(f'<text x="{x + 14}" y="{y + 40}" class="muted" style="font-size: 11.5px">{sub}</text>')
        gx0, gx1, gy0, gy1 = x + 48, x + pw - 50, y + ph - 30, y + 56
        values = [v for pts in t["traj"].values() for _, v in pts]
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.08
        ticks = nice_ticks(lo - pad, hi + pad, 4)
        vlo, vhi = min(ticks[0], lo - pad), max(ticks[-1], hi + pad)

        def fx(h: float, gx0=gx0, gx1=gx1, budget=budget) -> float:
            return gx0 + (gx1 - gx0) * h / budget

        def fy(v: float, gy0=gy0, gy1=gy1, vlo=vlo, vhi=vhi) -> float:
            return gy0 - (gy0 - gy1) * (v - vlo) / (vhi - vlo)

        for v in ticks:
            if vlo <= v <= vhi:
                out.append(f'<path d="M{gx0} {fy(v):.1f} H{gx1}" style="stroke: var(--grid); stroke-width: 1"/>'
                           f'<text x="{gx0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end" class="muted" '
                           f'style="font-size: 11.5px">{v:g}</text>')
        for h in (0, budget // 2, budget):
            out.append(f'<text x="{fx(h):.1f}" y="{gy0 + 18}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
                       f'{h} h</text>')
        for key_ in ("glm", "deepseek", "qwen", "sol", "astra", "claude"):
            pts = t["traj"][key_]
            width = 2.6 if key_ in AGENT_COLOR else 1.5
            out.append(drawn_path(tl, step_points(pts, budget, fx, fy), gx0, gx1, play * 0.06, play,
                                  f"fill: none; stroke: var({agent_color(key_)}); stroke-width: {width}; "
                                  f"stroke-linejoin: round; stroke-linecap: round"))
        winner = max(t["traj"], key=lambda k: t["traj"][k][-1][1])
        best = t["traj"][winner][-1][1]
        lab = tl.anim(hold_from(tl, play * 0.06 + (play - play * 0.06), "opacity: 1", "opacity: 0", 0.3))
        out.append(f'<g class="{lab}"><circle cx="{gx1}" cy="{fy(best):.1f}" r="4" style="fill: var({agent_color(winner)})"/>'
                   f'<text x="{gx1 + 8}" y="{fy(best) + 4:.1f}" style="font-size: 12px; font-weight: 700">{best:.0f}</text></g>')

    # Rank score: within each task the six runs are ranked; 100 means first everywhere, 0 last everywhere.
    lx0, lw = 660, 316
    out.append(f'<text x="{lx0}" y="{88 + 18}" style="font-size: 13.5px; font-weight: 650">rank score over 25 tasks</text>')
    out.append(f'<text x="{lx0}" y="{88 + 36}" class="muted" style="font-size: 12px">100 = first on every task, '
               f'0 = last on every task</text>')
    models = sorted(DATA["models"], key=lambda m: -m["rank_score"])
    for i, m in enumerate(models):
        y = 88 + 58 + 52 * i
        out.append(f'<text x="{lx0}" y="{y + 12}" style="font-size: 12.5px">{m["name"]}</text>')
        wins = f'{m["wins"]} win' + ("" if m["wins"] == 1 else "s")
        out.append(f'<text x="{lx0 + lw}" y="{y + 12}" text-anchor="end" style="font-size: 12.5px">'
                   f'<tspan style="font-weight: 700">{m["rank_score"]:.1f}</tspan>'
                   f'<tspan class="muted"> · {wins}</tspan></text>')
        out.append(f'<rect x="{lx0}" y="{y + 20}" width="{lw}" height="12" rx="6" style="fill: var(--faint)"/>')
        grow = tl.anim([(0, "transform: scaleX(0)"), (0.2 + 0.12 * i, "transform: scaleX(0)"),
                        (1.2 + 0.12 * i, f"transform: scaleX({m['rank_score'] / 100:.4f})"),
                        (T - 0.4, f"transform: scaleX({m['rank_score'] / 100:.4f})"), (T, "transform: scaleX(0)")],
                       "transform-box: fill-box; transform-origin: left center;")
        out.append(f'<rect x="{lx0}" y="{y + 20}" width="{lw}" height="12" rx="6" class="{grow}" '
                   f'style="fill: var({agent_color(m["key"])})"/>')
    s_ = DATA["summary"]
    out.append(f'<text x="{lx0}" y="{88 + 58 + 52 * 6 + 14}" class="muted" style="font-size: 12px">'
               f'{s_["runs"]} runs, one per agent and task · {s_["submissions"]:,} submissions</text>')
    out.append(f'<text x="{lx0}" y="{88 + 58 + 52 * 6 + 32}" class="muted" style="font-size: 12px">'
               f'{s_["gpu_hours"]} GPU-hours on H100s</text>')
    return svg(W, H, "Six agents on Infrathon: best valid score over the time budget on four tasks, and the rank score "
                     "over all 25 tasks", tl.css(), "\n".join(out))


# ---- 4. the 25 tasks -------------------------------------------------------------------------------------

def tasks_map() -> str:
    W = 1000
    out: list[str] = []
    out.append(header("25 tasks", "real systems, from GPU kernels to end-to-end training and serving"))
    lx = 790
    for stage, label in (("train", "training"), ("infer", "inference")):
        out.append(f'<rect x="{lx}" y="28" width="14" height="14" rx="3" style="fill: var(--{stage}-soft); '
                   f'stroke: var(--{stage})"/><text x="{lx + 21}" y="39.5" style="font-size: 12.5px">{label}</text>')
        lx += 96
    levels = [("operator", "Operators", "GPU kernels and", "communication primitives"),
              ("module", "Modules", "one part of a training", "or inference stack"),
              ("system", "Systems", "end-to-end training", "loops and serving")]
    cols, chip_w, chip_h, gap = 3, 248, 30, 8
    cx0 = 212
    y = 60
    for level, title, d1, d2 in levels:
        items = [t for t in DATA["tasks"] if t["level"] == level]
        items.sort(key=lambda t: t["stage"] != "training")
        rows = math.ceil(len(items) / cols)
        bh = rows * (chip_h + gap) - gap + 28
        budgets = sorted({t["budget_h"] for t in items})
        out.append(f'<rect x="24" y="{y}" width="{W - 48}" height="{bh}" rx="14" class="card"/>')
        out.append(f'<text x="42" y="{y + 32}" style="font-size: 16px; font-weight: 650">{title}</text>')
        out.append(f'<text x="42" y="{y + 52}" style="font-size: 12.5px"><tspan style="font-weight: 700">{len(items)}</tspan>'
                   f'<tspan class="muted"> tasks · {"/".join(str(b) for b in budgets)} h each</tspan></text>')
        if rows > 1:
            out.append(f'<text x="42" y="{y + 72}" class="muted" style="font-size: 11.5px">{d1}</text>')
            out.append(f'<text x="42" y="{y + 87}" class="muted" style="font-size: 11.5px">{d2}</text>')
        else:
            out.append(f'<text x="42" y="{y + 72}" class="muted" style="font-size: 11.5px">{d1} {d2}</text>')
        for i, t in enumerate(items):
            r, c = divmod(i, cols)
            x, yy = cx0 + c * (chip_w + gap), y + 14 + r * (chip_h + gap)
            stage = "train" if t["stage"] == "training" else "infer"
            out.append(f'<g><rect x="{x}" y="{yy}" width="{chip_w}" height="{chip_h}" rx="8" '
                       f'style="fill: var(--{stage}-soft); stroke: var(--{stage}); stroke-opacity: .35"/>'
                       f'<rect x="{x + 7}" y="{yy + 8}" width="3.5" height="{chip_h - 16}" rx="1.75" style="fill: var(--{stage})"/>'
                       f'<text x="{x + 19}" y="{yy + 19.5}" style="font-size: 12.5px">{esc(t["name"])}</text></g>')
        y += bh + 10
    H = y + 8
    return svg(W, H, "The 25 Infrathon tasks by level (operators, modules, systems) and stage (training, inference)",
               "", "\n".join(out))


if __name__ == "__main__":
    for name, make in (("overview.svg", overview), ("discovery.svg", discovery), ("results.svg", results),
                       ("tasks.svg", tasks_map)):
        (HERE / name).write_text(make(), encoding="utf-8")
        print(f"wrote {HERE / name}")
