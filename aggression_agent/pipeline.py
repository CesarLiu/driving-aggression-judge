"""The run: screen every scene, pick the ones worth a look, review them with agents, write labels.

1. **Screen** (no model): the skill's ``screen_crossing.py`` and ``screen_same_direction.py`` over the
   whole folder, or their earlier output passed in.
2. **Triage** (no model): a scene goes to review when a screen found a right-of-way violation or a
   same-direction candidate, when a screen note marks a borderline or doubtful case, or when the two
   cars end up in one lane after starting side by side without the screen having seen a lane change
   (the lane-branch blind spot). Every other scene keeps the screens' verdict: not aggressive.
3. **Review**: one reviewer agent per triaged scene.
4. **Critique**: a critic agent where the reviewer overruled the screens, was unsure (low confidence)
   or asked for a second look; its verdict is final.
5. **Labels**: ``labels.csv`` (every screened scene), ``disagreements.csv`` (final verdict differs from
   the screens), ``report.md``; ``reviews/<scene>.json`` keeps each agent's verdict, tool calls and
   cost, and ``figures/`` the figures the agents looked at.

Runs resume: screens and reviews already in the run folder are reused.
"""

import asyncio
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import scene_tools as T


def load_backend(args):
    """The agent backend: Claude through the Claude Agent SDK, or an OpenAI-compatible server (vLLM, Ollama)."""
    if args.backend == "claude":
        import agents as backend
    else:
        import local_agents as backend

        backend.configure(base_url=args.base_url, api_key=args.api_key, images=not args.no_images,
                          max_images=args.max_images, timeout=args.request_timeout)
    return backend

DOUBT_NOTES = ("borderline_cut_in_thw", "cut_in_pulling_away", "implausible_lateral_move", "boxes_touch")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return {r["scene"]: r for r in csv.DictReader(f)}


def screen(args, run):
    scripts = Path(args.skill_dir) / "scripts"
    out = {}
    for name, script, given, extra in (
            ("crossing", "screen_crossing.py", args.crossing_csv, ["--profile", args.profile]),
            ("same_direction", "screen_same_direction.py", args.same_direction_csv, [])):
        path = Path(given) if given else run / ("screen_%s.csv" % name)
        if not path.exists():
            cmd = [sys.executable, str(scripts / script), "--scenes", str(args.scenes), "--out", str(path),
                   "--pairs", args.pairs] + extra
            print("screening:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        out[name] = read_csv(path)
        print("%s screen: %d scenes from %s" % (name, len(out[name]), path), flush=True)
    return out


def one(v):
    return str(v).strip() in ("1", "1.0", "True", "true")


def screen_finding(cross, same, side):
    """What the screens say about one side ('sdc' = first, 'ooi' = second): '' when nothing."""
    found = []
    if same and same.get(side + "_kind"):
        found.append(same[side + "_kind"])
    if cross and one(cross.get(side + "_aggressive_candidate")):
        found.append("right_of_way")
    return "+".join(found)


def ended_in_one_lane(scene, a, b):
    """True when the two start side by side in neighbouring lanes (2.3-5.5 m apart sideways, within
    40 m, heading the same way) and later one is right behind the other (under 1 m sideways)."""
    both = np.flatnonzero(scene.valid[a] & scene.valid[b])
    if both.size < 20:
        return False

    def lateral(k):
        if abs(T.wrap_deg(math.degrees(scene.heading[b, k] - scene.heading[a, k]))) > 20:
            return None
        ahead, left = T.rel(scene, a, b, k)
        return abs(left) if abs(ahead) < 40 else None

    start = [lateral(k) for k in both[:10]]
    if not any(x is not None and 2.3 <= x <= 5.5 for x in start):
        return False
    return any(x is not None and x < 1.0 for x in (lateral(k) for k in both[10:]))


def triage(screens, scene_paths, blind_spot_scan=True):
    cross, same = screens["crossing"], screens["same_direction"]
    picked = {}
    for name in scene_paths:
        c, s = cross.get(name), same.get(name)
        why = []
        if c:
            for side, who in (("sdc", "first"), ("ooi", "second")):
                if one(c.get(side + "_violation")):
                    why.append("crossing screen: %s went first against the other's right of way (%s; affected=%s %s)"
                               % (who, c.get("case"), c.get("affected") or "unknown", c.get("affected_by") or ""))
        if s:
            for side, who in (("sdc", "first"), ("ooi", "second")):
                if s.get(side + "_kind"):
                    why.append("same-direction screen: %s %s" % (who, s[side + "_kind"]))
                notes = s.get(side + "_notes") or ""
                doubt = [n for n in notes.split(";") if n.startswith(DOUBT_NOTES)]
                if doubt:
                    why.append("same-direction screen note on the %s: %s" % (who, ";".join(doubt)))
            notes = (s.get("sdc_notes") or "") + (s.get("ooi_notes") or "")
            if "lane_join" in notes and "cut_in" not in (s.get("sdc_kind") or "") + (s.get("ooi_kind") or ""):
                # the lane-branch blind spot: one car joined the other's lane, but measured against its
                # own lane chain neither moved sideways, so neither got a cut-in (CAT scenes 15, 69, 156, 293, ...)
                why.append("same-direction screen saw one car join the other's lane but credited the move to "
                           "neither (lane branch or merge on the map?): who moved, and with what gap?")
        if why:
            picked[name] = why
    if blind_spot_scan:
        for name, path in scene_paths.items():
            s = same.get(name)
            if name in picked or not s:
                continue
            try:
                scene, _ = T._HELPERS["load_scene"](path, T._HELPERS["Scene"])
                a, b = scene.index(s["first_id"]), scene.index(s["second_id"])
            except sys.modules["scenes"].NeedsTorch as exc:
                sys.exit(str(exc))
            except Exception as exc:
                print("blind-spot scan skips %s: %s" % (name, exc), file=sys.stderr)
                continue
            if ended_in_one_lane(scene, a, b):
                picked[name] = ["the two start side by side and end up one behind the other, but the "
                                "same-direction screen saw no lane change (lane branch or merge?)"]
    return picked


def disagrees(verdict, cross, same):
    for key, side in (("first", "sdc"), ("second", "ooi")):
        if bool(screen_finding(cross, same, side)) != verdict[key]["aggressive"]:
            return True
    return False


async def review_all(names, screens, args, run, rules, agents):
    sem = asyncio.Semaphore(args.concurrency)
    reviews = run / "reviews"
    reviews.mkdir(exist_ok=True)
    fig_dir = str(run / "figures")

    async def one_scene(name):
        path = reviews / ("%s.json" % name)
        if path.exists():
            done = json.loads(path.read_text(encoding="utf-8"))
            if done.get("final"):
                return name, done
        async with sem:
            try:
                return name, await review_one(name, path)
            except Exception as exc:  # one failed agent must not cancel the others' paid work
                print("%s: agent run failed: %s: %s" % (name, type(exc).__name__, exc), file=sys.stderr)
                rec = {"scene": name, "final": None, "error": "%s: %s" % (type(exc).__name__, exc)}
                path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                return name, rec

    async def review_one(name, path):
        rec = {"scene": name}
        r = await agents.review(name, rules, args.reviewer_model, fig_dir, args.budget, args.max_turns)
        rec["reviewer"] = r
        final, source = r["verdict"], "reviewer"
        c, s = screens["crossing"].get(name), screens["same_direction"].get(name)
        if final is None:
            print("%s: reviewer gave no verdict (%s)" % (name, r["subtype"]), file=sys.stderr)
        want = final is not None and args.critic != "none" and (
            args.critic == "all" or final["needs_second_look"] or disagrees(final, c, s)
            or "low" in (final["first"]["confidence"], final["second"]["confidence"]))
        if want:
            k = await agents.criticise(name, final, rules, args.critic_model, fig_dir, args.budget * 1.5, args.max_turns)
            rec["critic"] = k
            if k["verdict"] is not None:
                final, source = k["verdict"], "critic"
        rec["final"], rec["source"] = final, source
        rec["cost_usd"] = r["cost_usd"] + rec.get("critic", {}).get("cost_usd", 0.0)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print("%s: %s first=%s second=%s ($%.2f)" % (
            name, source, final and final["first"]["aggressive"], final and final["second"]["aggressive"],
            rec["cost_usd"]), flush=True)
        return rec

    return dict(await asyncio.gather(*(one_scene(n) for n in names)))


COLUMNS = ["scene", "scenario_id", "format", "pair", "first_id", "second_id", "jurisdiction",
           "sdc_aggressive", "sdc_kind", "sdc_reason", "sdc_confidence",
           "ooi_aggressive", "ooi_kind", "ooi_reason", "ooi_confidence",
           "source", "screen_sdc", "screen_ooi", "triage"]


def write_outputs(run, screens, scene_paths, picked, reviews, profile):
    cross, same = screens["crossing"], screens["same_direction"]
    rows, cost = [], 0.0
    for name in scene_paths:
        c, s = cross.get(name), same.get(name)
        base = c or s
        if base is None:
            continue
        row = {k: base.get(k, "") for k in ("scene", "scenario_id", "format", "pair", "first_id", "second_id")}
        row.update(jurisdiction=profile["id"], screen_sdc=screen_finding(c, s, "sdc"),
                   screen_ooi=screen_finding(c, s, "ooi"), triage=" | ".join(picked.get(name, [])))
        rec = reviews.get(name)
        if rec and rec.get("final"):
            v = rec["final"]
            cost += rec.get("cost_usd", 0.0)
            for key, side in (("first", "sdc"), ("second", "ooi")):
                row.update({side + "_aggressive": int(v[key]["aggressive"]), side + "_kind": v[key]["kind"],
                            side + "_reason": v[key]["reason"], side + "_confidence": v[key]["confidence"]})
            row["source"] = rec["source"]
        else:
            for side in ("sdc", "ooi"):
                f = row["screen_" + side]
                row.update({side + "_aggressive": int(bool(f)), side + "_kind": f or "none",
                            side + "_reason": "" if f else "no screen finding",
                            side + "_confidence": "" if f else "screen"})
            row["source"] = "screen" if name not in picked else "not_reviewed"
        rows.append(row)
    with open(run / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    diff = [r for r in rows if r["source"] in ("reviewer", "critic")
            and (bool(r["screen_sdc"]) != bool(r["sdc_aggressive"]) or bool(r["screen_ooi"]) != bool(r["ooi_aggressive"]))]
    with open(run / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(diff)

    reviewed = [r for r in rows if r["source"] in ("reviewer", "critic")]
    pos = [r for r in rows if r["sdc_aggressive"] or r["ooi_aggressive"]]
    lines = ["# Aggression labels: %s" % run.name, "",
             "Jurisdiction: %s. Scenes screened: %d; sent for review: %d; reviewed: %d (critic final in %d); "
             "agent cost $%.2f." % (profile["id"], len(rows), len(picked), len(reviewed),
                                    sum(r["source"] == "critic" for r in rows), cost), "",
             "First = `sdc_*` columns, second = `ooi_*` (the pair the screens judged; `pair` says how it was chosen).",
             "", "## Aggressive (%d scenes)" % len(pos), ""]
    for r in pos:
        for side, who in (("sdc", "first"), ("ooi", "second")):
            if r[side + "_aggressive"]:
                lines.append("- **%s** %s %s (%s, %s): %s" % (r["scene"], who, r[side + "_kind"], r["source"],
                                                             r[side + "_confidence"], r[side + "_reason"]))
    lines += ["", "## Final verdict differs from the screens (%d scenes)" % len(diff), ""]
    for r in diff:
        lines.append("- **%s** screens: first `%s`, second `%s` → final: first %s, second %s (%s). %s %s" % (
            r["scene"], r["screen_sdc"] or "-", r["screen_ooi"] or "-", r["sdc_kind"] if r["sdc_aggressive"] else "none",
            r["ooi_kind"] if r["ooi_aggressive"] else "none", r["source"], r["sdc_reason"], r["ooi_reason"]))
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s, %s, %s; %d aggressive scenes, %d disagreements with the screens, agent cost $%.2f" % (
        run / "labels.csv", run / "disagreements.csv", run / "report.md", len(pos), len(diff), cost))


async def run_pipeline(args):
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    T.setup(args.skill_dir)
    sys.path.insert(0, str(Path(args.skill_dir) / "scripts"))
    import jurisdiction

    profile = jurisdiction.load_profile(args.profile)
    if profile["status"].startswith("draft"):
        print("WARNING: profile %s is a draft; its verdicts are unvalidated" % profile["id"], file=sys.stderr)
    paths = {p.stem: p for p in T.scene_files(args.scenes)}
    if args.only:
        keep = set(args.only.split(","))
        paths = {k: v for k, v in paths.items() if k in keep}
    screens = screen(args, run)
    picked = triage(screens, paths, not args.no_blind_spot_scan)
    names = list(picked)[: args.max_review] if args.max_review is not None else list(picked)
    print("%d scenes, %d sent for review%s" % (len(paths), len(picked),
                                               ", reviewing the first %d" % len(names) if len(names) < len(picked) else ""))
    (run / "triage.json").write_text(json.dumps(picked, indent=1), encoding="utf-8")
    for name in names:
        c, s = screens["crossing"].get(name), screens["same_direction"].get(name)
        base = s or c
        T.register(name, paths[name], base["first_id"], base["second_id"], base["pair"],
                   {k: v for k, v in (("crossing", c), ("same_direction", s)) if v}, picked[name])
    reviews = {}
    if names and not args.screen_only:
        agents = load_backend(args)
        rules = agents.rules_text(args.skill_dir, profile)
        reviews = await review_all(names, screens, args, run, rules, agents)
    write_outputs(run, screens, paths, picked, reviews, profile)
