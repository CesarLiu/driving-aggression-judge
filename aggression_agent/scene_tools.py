"""What the reviewing agents can look at in a scene: an overview of the judged pair, a time series of
their states and the gap between them, the other road users near them, and a figure.

Scenes are read with the judge-driving-aggression skill's loader (``scripts/scenes.py``), so both
formats the screens accept work here: ScenarioNet / MetaDrive pickles and catk's cached WOMD. The
pair is the one the screens judged (``first_id``, ``second_id`` of their rows); in the screens' CSVs
``sdc_*`` columns belong to the first agent and ``ooi_*`` to the second, whoever they are.

Speeds are the logged velocity's norm. Braking is speed lost over the next 1 s per second (the
screens' ``sustained_decel``), because WOMD speeds jump by 1-2 m/s between frames.
"""

import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DT = 0.1
_HELPERS = {}


def setup(skill_dir):
    """Put the skill's scripts (with their copy of the ``responsibility`` package) on the path and
    import what the tools reuse."""
    scripts = Path(skill_dir) / "scripts"
    if not (scripts / "scenes.py").exists():
        sys.exit("no judge-driving-aggression skill at %s (pass --skill-dir)" % skill_dir)
    sys.path.insert(0, str(scripts))
    import screen_crossing  # noqa: E402
    import screen_same_direction  # noqa: E402
    import scenes  # noqa: E402

    from responsibility.scene import Scene  # noqa: E402

    _HELPERS.update(Scene=Scene, load_scene=scenes.load_scene, sustained_decel=screen_crossing.sustained_decel,
                    smooth=screen_crossing.smooth, corners=screen_same_direction.corners,
                    box_gap=screen_same_direction.box_gap, scene_files=screen_crossing.scene_files)


def scene_files(path):
    return _HELPERS["scene_files"](path)


@dataclass
class Entry:
    name: str
    path: Path
    scene: object
    fmt: str
    a: int  # first agent of the pair
    b: int  # second
    pair: str
    screens: dict = field(default_factory=dict)  # screen name -> its CSV row for this scene
    triage: list = field(default_factory=list)  # why the scene was sent for review


_SCENES = {}


def register(name, path, first_id, second_id, pair, screens, triage=()):
    scene, fmt = _HELPERS["load_scene"](path, _HELPERS["Scene"])
    _SCENES[name] = Entry(name, Path(path), scene, fmt, scene.index(first_id), scene.index(second_id), pair,
                          dict(screens), list(triage))
    return _SCENES[name]


def entry(name):
    if name not in _SCENES:
        raise KeyError("scene %r is not loaded; loaded: %s" % (name, ", ".join(sorted(_SCENES)) or "none"))
    return _SCENES[name]


# --- measurements ----------------------------------------------------------------------------

def speed(scene, i):
    return np.linalg.norm(scene.velocity[i, :, :2], axis=1)


def role(scene, i):
    tags = []
    if i == scene.sdc:
        tags.append("SDC")
    if i in scene.objects_of_interest:
        tags.append("object of interest")
    return "track %s, %s%s" % (scene.track_ids[i], scene.types[i].lower(), (" (" + ", ".join(tags) + ")") if tags else "")


def name_of(e, i):
    if i == e.scene.sdc:
        return "SDC"
    return "first" if i == e.a else "second" if i == e.b else "track " + e.scene.track_ids[i]


def box(scene, i, k):
    length, width = scene.shape_at(i, k)
    return _HELPERS["corners"](scene.position[i, k, :2].astype(float), float(scene.heading[i, k]), float(length), float(width))


def rel(scene, i, j, k):
    """Position of j's centre in i's frame at step k: (ahead, left) in metres."""
    d = scene.position[j, k, :2] - scene.position[i, k, :2]
    h = float(scene.heading[i, k])
    return float(d[0] * math.cos(h) + d[1] * math.sin(h)), float(-d[0] * math.sin(h) + d[1] * math.cos(h))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def stops(v, valid, below=0.5):
    """(start s, duration s) of every stop (speed below 0.5 m/s) of at least 0.3 s."""
    out, k = [], 0
    while k < len(v):
        if valid[k] and v[k] < below:
            j = k
            while j < len(v) and valid[j] and v[j] < below:
                j += 1
            if j - k >= 3:
                out.append((k * DT, (j - k) * DT))
            k = j
        else:
            k += 1
    return out


def pair_state(scene, a, b, k):
    """Box gap and, when one follows the other in its path, who leads and the follower's headway/TTC."""
    gap = _HELPERS["box_gap"](box(scene, a, k), box(scene, b, k))
    out = {"gap": gap}
    for f, l in ((a, b), (b, a)):
        ahead, left = rel(scene, f, l, k)
        lf, wf = scene.shape_at(f, k)
        ll, wl = scene.shape_at(l, k)
        same_way = abs(wrap_deg(math.degrees(scene.heading[l, k] - scene.heading[f, k]))) < 30
        if ahead > 0 and same_way and abs(left) < (wf + wl) / 2:
            vf, vl = speed(scene, f)[k], speed(scene, l)[k]
            room = ahead - (lf + ll) / 2
            out.update(follower=f, leader=l, room=room, thw=room / vf if vf > 0.5 else None,
                       ttc=room / (vf - vl) if vf - vl > 0.1 and room > 0 else None)
            break
    return out


# --- tool bodies (plain functions returning text; the MCP wrappers are in agents.py) -----------

def overview(name):
    e = entry(name)
    s, a, b = e.scene, e.a, e.b
    lines = ["Scene %s (%s format), scenario %s, 9.1 s at 10 Hz (steps 0-90)." % (name, e.fmt, s.scenario_id),
             "Pair (%s): first = %s; second = %s." % (e.pair, role(s, a), role(s, b))]
    if e.fmt == "catk_cache":
        lines.append("catk cache: the map is rebuilt from SMART tokens (lane links inferred from geometry, stop "
                     "signs at the start of their lane, driveways missing), lights known only at 1.0 s.")
    for label, i in (("first", a), ("second", b)):
        v, valid = speed(s, i), s.valid[i]
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            lines.append("%s: never valid" % label)
            continue
        dec = _HELPERS["sustained_decel"](np.where(valid, v, np.nan))
        k_dec = int(np.nanargmax(dec)) if np.isfinite(dec).any() else None
        turn = wrap_deg(math.degrees(s.heading[i, idx[-1]] - s.heading[i, idx[0]]))
        dist = float(np.linalg.norm(np.diff(s.position[i, idx, :2], axis=0), axis=1).sum())
        length, width = s.shape_at(i, int(idx[0]))
        at = ", ".join("%g s %.1f" % (t, v[int(t / DT)]) if valid[int(t / DT)] else "%g s -" % t
                       for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        st = stops(v, valid)
        lines.append("%s (%s): %.1f x %.1f m, valid %.1f-%.1f s, travelled %.0f m, heading change %+.0f deg "
                     "(left positive).\n  speed m/s at %s.\n  stops: %s. Hardest braking held 1 s: %s." % (
                         label, role(s, i), length, width, idx[0] * DT, idx[-1] * DT, dist, turn, at,
                         ", ".join("from %.1f s for %.1f s" % x for x in st) or "none",
                         "%.1f m/s^2 from %.1f s" % (dec[k_dec], k_dec * DT) if k_dec is not None else "n/a"))
    both = np.flatnonzero(s.valid[a] & s.valid[b])
    if both.size:
        gaps = [pair_state(s, a, b, k) for k in both]
        g = np.array([x["gap"] for x in gaps])
        k = int(np.argmin(g))
        lines.append("Box-to-box gap: min %.1f m at %.1f s; %.1f m at %.1f s, %.1f m at %.1f s." % (
            g[k], both[k] * DT, g[0], both[0] * DT, g[-1], both[-1] * DT))
        for k0 in (both[0], both[-1]):
            ahead, left = rel(s, a, b, k0)
            lines.append("At %.1f s the second is %.1f m ahead, %.1f m left of the first; headings differ by %+.0f deg." % (
                k0 * DT, ahead, left, wrap_deg(math.degrees(s.heading[b, k0] - s.heading[a, k0]))))
        fol = [x for x in gaps if "thw" in x and x["thw"] is not None]
        if fol:
            x = min(fol, key=lambda x: x["thw"])
            kk = both[gaps.index(x)]
            lines.append("Following in path: shortest headway %.2f s at %.1f s (%s behind %s, %.1f m)." % (
                x["thw"], kk * DT, name_of(e, x["follower"]), name_of(e, x["leader"]), x["room"]))
    else:
        lines.append("The two are never valid at the same step.")
    for screen, row in e.screens.items():
        filled = {k: v for k, v in row.items() if v not in ("", None) and k not in ("scene", "scenario_id", "format", "first_id", "second_id")}
        lines.append("%s screen row (sdc_* = first, ooi_* = second): %s" % (screen, "; ".join("%s=%s" % kv for kv in filled.items())))
    if e.triage:
        lines.append("Sent for review because: " + "; ".join(e.triage))
    return "\n".join(lines)


def timeseries(name, start=0.0, end=9.0, every=0.5):
    e = entry(name)
    s, a, b = e.scene, e.a, e.b
    every = max(float(every), 0.1)
    steps = sorted(set(int(round(t / DT)) for t in np.arange(max(start, 0), min(end, 9.0) + 1e-6, every)))
    va, vb = speed(s, a), speed(s, b)
    da = _HELPERS["sustained_decel"](va)
    db = _HELPERS["sustained_decel"](vb)
    head = ("t_s | first x y v_mps heading_deg brake_1s | second x y v_mps heading_deg brake_1s | gap_m | "
            "second_rel_first ahead_m left_m | follower leader room_m thw_s ttc_s")
    out = [head]
    for k in steps:
        if k >= s.n_steps:
            break
        cells = ["%.1f" % (k * DT)]
        for i, v, d in ((a, va, da), (b, vb, db)):
            if s.valid[i, k]:
                cells.append("%.1f %.1f %.1f %.0f %s" % (s.position[i, k, 0], s.position[i, k, 1], v[k],
                                                          math.degrees(s.heading[i, k]),
                                                          "%.1f" % d[k] if np.isfinite(d[k]) else "-"))
            else:
                cells.append("invalid")
        if s.valid[a, k] and s.valid[b, k]:
            st = pair_state(s, a, b, k)
            cells.append("%.1f" % st["gap"])
            cells.append("%.1f %.1f" % rel(s, a, b, k))
            if "follower" in st:
                cells.append("%s %s %.1f %s %s" % (name_of(e, st["follower"]), name_of(e, st["leader"]), st["room"],
                                                   "%.2f" % st["thw"] if st["thw"] is not None else "-",
                                                   "%.1f" % st["ttc"] if st["ttc"] is not None else "-"))
            else:
                cells.append("not following")
        else:
            cells.append("-")
        out.append(" | ".join(cells))
    return "\n".join(out)


def nearby(name, t, radius=30.0):
    e = entry(name)
    s, a, b = e.scene, e.a, e.b
    k = min(max(int(round(t / DT)), 0), s.n_steps - 1)
    rows = []
    for i in range(s.n_agents):
        if i in (a, b) or not s.valid[i, k]:
            continue
        dists = [float(np.linalg.norm(s.position[i, k, :2] - s.position[j, k, :2])) for j in (a, b) if s.valid[j, k]]
        if not dists or min(dists) > radius:
            continue
        ref = a if s.valid[a, k] else b
        ahead, left = rel(s, ref, i, k)
        parked = speed(s, i)[s.valid[i]].max() <= 0.5
        rows.append((parked, min(dists), "%s: %.1f m ahead, %.1f m left of the %s, %.1f m/s, heading %+.0f deg relative" % (
            role(s, i), ahead, left, "first" if ref == a else "second", speed(s, i)[k],
            wrap_deg(math.degrees(s.heading[i, k] - s.heading[ref, k])))))
    rows.sort()
    moving = [r[2] for r in rows if not r[0]][:12]
    parked = [r[2] for r in rows if r[0]]
    text = "Road users within %.0f m of the pair at %.1f s, nearest first:\n" % (radius, k * DT)
    text += "\n".join(moving) if moving else "no moving ones"
    if parked:
        text += "\n%d never move in the clip (parked or queued); nearest: %s" % (len(parked), "; ".join(parked[:4]))
    return text


def render(name, marks=None, zoom_at=None, out_dir=None):
    """A PNG: the map around the pair with both tracks (dots every 1 s) and their boxes at the marked
    times, other road users' tracks in grey; beside it speed and box gap over time. ``zoom_at`` (s)
    shows 60 m around the pair at that time instead of the whole of both tracks. Returns bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    e = entry(name)
    s, a, b = e.scene, e.a, e.b
    if not marks:
        marks = [0.0, 3.0, 6.0, 9.0]
        for row in e.screens.values():
            for key in ("sdc_cut_in_t", "ooi_cut_in_t", "sdc_close_pass_t", "ooi_close_pass_t"):
                if row.get(key) not in ("", None):
                    marks.append(float(row[key]))
    marks = sorted(set(round(min(max(float(m), 0.0), 9.0), 1) for m in marks))
    if zoom_at is not None:
        k = min(max(int(round(float(zoom_at) / DT)), 0), s.n_steps - 1)
        here = [s.position[i, k, :2] for i in (a, b) if s.valid[i, k]] or [s.position[a, s.valid[a], :2].mean(0)]
        mid, span = np.mean(here, axis=0), 60.0
        marks = sorted(set(marks) | {round(k * DT, 1)})
    else:
        pts = np.concatenate([s.position[i, s.valid[i], :2] for i in (a, b)])
        lo, hi = pts.min(0) - 20, pts.max(0) + 20
        span, mid = max(hi - lo), (lo + hi) / 2
    lo, hi = mid - span / 2, mid + span / 2

    fig = plt.figure(figsize=(15, 7.5), dpi=90)
    ax = fig.add_axes([0.03, 0.06, 0.55, 0.88])
    for f in s.map_features.values():
        kind = str(f.get("type", ""))
        if "polyline" in f and len(f["polyline"]) >= 2:
            p = np.asarray(f["polyline"])[:, :2]
            if kind.startswith("LANE"):
                ax.plot(p[:, 0], p[:, 1], color="#c8c8c8", lw=0.8, zorder=1)
                ax.annotate("", p[-1], p[-2], arrowprops=dict(arrowstyle="->", color="#c8c8c8", lw=0.6), zorder=1)
            elif kind.startswith("ROAD_EDGE"):
                ax.plot(p[:, 0], p[:, 1], color="#555555", lw=1.0, zorder=1)
            elif kind.startswith("ROAD_LINE"):
                ax.plot(p[:, 0], p[:, 1], color="#d9c27a" if "YELLOW" in kind else "#dddddd", lw=0.7, zorder=1)
        elif "polygon" in f and len(f["polygon"]) >= 3:
            ax.add_patch(Polygon(np.asarray(f["polygon"])[:, :2], closed=True, fc="#eef3ff", ec="#b8c6e6", lw=0.5, zorder=0))
        if kind == "STOP_SIGN" and "position" in f:
            p = np.asarray(f["position"])
            ax.plot(p[0], p[1], marker="8", color="#d62728", ms=9, zorder=4)
    for i in range(s.n_agents):
        if i in (a, b) or not s.valid[i].any():
            continue
        p = s.position[i, s.valid[i], :2]
        if ((p > lo) & (p < hi)).all(1).any():
            ax.plot(p[:, 0], p[:, 1], color="#999999", lw=0.8, alpha=0.6, zorder=2)
            k0 = int(np.flatnonzero(s.valid[i])[0])
            ax.add_patch(Polygon(box(s, i, k0), closed=True, fc="#bbbbbb", ec="#888888", alpha=0.5, zorder=2))
            if speed(s, i)[s.valid[i]].max() > 0.5:
                ax.text(*s.position[i, k0, :2], s.track_ids[i], fontsize=6, color="#777777", zorder=3)
    colours = {a: "#1f77b4", b: "#ff7f0e"}
    for i in (a, b):
        valid = s.valid[i]
        p = s.position[i, valid, :2]
        ax.plot(p[:, 0], p[:, 1], color=colours[i], lw=1.5, zorder=5,
                label="%s: %s" % ("first" if i == a else "second", role(s, i)))
        for k in range(0, s.n_steps, 10):
            if valid[k]:
                ax.plot(*s.position[i, k, :2], "o", color=colours[i], ms=3, zorder=6)
                ax.text(*s.position[i, k, :2], " %d" % (k // 10), fontsize=7, color=colours[i], zorder=6)
        for n, t in enumerate(marks):
            k = int(round(t / DT))
            if k < s.n_steps and valid[k]:
                alpha = 0.25 + 0.6 * (n + 1) / len(marks)
                ax.add_patch(Polygon(box(s, i, k), closed=True, fc=colours[i], ec="black", lw=0.6, alpha=alpha, zorder=7))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(("zoomed at %g s; " % float(zoom_at) if zoom_at is not None else "") + "scene %s: tracks with a dot every 1 s (number = second), boxes at %s s; red octagon = stop sign"
                 % (name, ", ".join("%g" % m for m in marks)), fontsize=9)

    t = np.arange(s.n_steps) * DT
    ax2 = fig.add_axes([0.64, 0.55, 0.34, 0.38])
    for i in (a, b):
        v = np.where(s.valid[i], speed(s, i), np.nan)
        ax2.plot(t, v, color=colours[i], label="first" if i == a else "second")
    ax2.set_ylabel("speed m/s")
    ax2.legend(fontsize=8)
    ax3 = fig.add_axes([0.64, 0.08, 0.34, 0.38], sharex=ax2)
    both = s.valid[a] & s.valid[b]
    gap = np.full(s.n_steps, np.nan)
    for k in np.flatnonzero(both):
        gap[k] = pair_state(s, a, b, k)["gap"]
    ax3.plot(t, gap, color="black")
    ax3.set_ylabel("box-to-box gap m")
    ax3.set_xlabel("time s")
    for axis in (ax2, ax3):
        axis.grid(alpha=0.3)
        for m in marks:
            axis.axvline(m, color="#888888", ls=":", lw=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = buf.getvalue()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / ("%s.png" % name)).write_bytes(data)
    return data
