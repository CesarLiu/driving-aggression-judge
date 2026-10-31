"""What every agent backend shares: the verdict schemas, the prompts, the scene tools as plain
specifications, and the check that keeps the rule auditor inside the skill and the run folder.

A backend (``agents.py`` for Claude through the Claude Agent SDK, ``local_agents.py`` for an
OpenAI-compatible server such as vLLM or Ollama) turns these into its own tool and prompt formats and
provides ``review``, ``criticise`` and ``audit`` with the same signatures and return records.

The rules the agents judge by are read at start-up from the judge-driving-aggression skill
(``references/rubric.md``, the "Rules of judgement" of ``SKILL.md`` and the jurisdiction profile), so
the framework and the skill cannot drift apart.
"""

import json
from pathlib import Path

import scene_tools as T

KINDS = ["none", "cut_in", "tailgate", "close_pass", "right_of_way", "other"]
SIDE = {
    "type": "object",
    "properties": {
        "aggressive": {"type": "boolean"},
        "kind": {"type": "string", "enum": KINDS},
        "reason": {"type": "string", "description": "one English sentence with the deciding rule and numbers"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["aggressive", "kind", "reason", "confidence"],
    "additionalProperties": False,
}
VERDICT = {
    "type": "object",
    "properties": {
        "first": SIDE,
        "second": SIDE,
        "screen_agreement": {"type": "string", "enum": ["agrees", "screen_false_positive", "screen_missed",
                                                       "screen_wrong_details", "no_screen_finding"]},
        "screen_comment": {"type": "string", "description": "what the screens got wrong or could not see, if anything"},
        "needs_second_look": {"type": "boolean"},
    },
    "required": ["first", "second", "screen_agreement", "screen_comment", "needs_second_look"],
    "additionalProperties": False,
}
CRITIC_VERDICT = {
    "type": "object",
    "properties": dict(VERDICT["properties"], upheld={"type": "boolean"},
                       critique={"type": "string", "description": "what in the reviewer's verdict did not hold, or why it holds"}),
    "required": VERDICT["required"] + ["upheld", "critique"],
    "additionalProperties": False,
}


def validate(value, schema, where="verdict"):
    """Errors of ``value`` against the small JSON-schema subset used here (empty list: valid)."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return ["%s must be an object" % where]
        errs = ["%s.%s is missing" % (where, k) for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errs += ["%s.%s is not allowed" % (where, k) for k in value if k not in schema["properties"]]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += validate(value[k], sub, "%s.%s" % (where, k))
        return errs
    ok = {"boolean": lambda v: isinstance(v, bool), "string": lambda v: isinstance(v, str),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "array": lambda v: isinstance(v, list)}.get(kind, lambda v: True)
    if not ok(value):
        return ["%s must be a %s" % (where, kind)]
    if "enum" in schema and value not in schema["enum"]:
        return ["%s must be one of %s" % (where, ", ".join(schema["enum"]))]
    return []


# --- scene tools ------------------------------------------------------------------------------
# Each tool is bound to the one scene an agent judges. ``call`` returns ("text", str) or
# ("image", png bytes).

SCENE_TOOLS = [
    {"name": "scene_overview",
     "description": "The judged pair (which is the SDC), both agents' speeds, stops, braking and turn, the gap "
                    "between them, the screens' rows for the scene and why it was sent for review. Start here.",
     "schema": {"type": "object", "properties": {}},
     "call": lambda scene, a, fig_dir: ("text", T.overview(scene))},
    {"name": "pair_timeseries",
     "description": "Both agents' position, speed, heading and braking held over 1 s, the box-to-box gap, the "
                    "second's position in the first's frame, and when one follows the other in its path the "
                    "follower, room, headway (thw) and time to collision, sampled from start to end (s) every "
                    "`every` s (min 0.1).",
     "schema": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                                                 "every": {"type": "number"}}},
     "call": lambda scene, a, fig_dir: ("text", T.timeseries(scene, a.get("start", 0.0), a.get("end", 9.0),
                                                              a.get("every", 0.5)))},
    {"name": "nearby_road_users",
     "description": "Other road users within `radius` m (default 30) of the pair at time t (s): position relative "
                    "to the first, speed, relative heading. Use it to see whether a third party explains a brake, "
                    "a stop or a lane change.",
     "schema": {"type": "object", "properties": {"t": {"type": "number"}, "radius": {"type": "number"}},
                "required": ["t"]},
     "call": lambda scene, a, fig_dir: ("text", T.nearby(scene, a["t"], a.get("radius", 30.0)))},
    {"name": "render_scene",
     "image": True,
     "description": "A figure: map (lanes grey with arrows, road edges dark, stop signs red octagons), the first "
                    "(blue) and second (orange) tracks with a dot every 1 s, their boxes at the `marks` times (s), "
                    "other road users' tracks in grey; speed and box gap over time beside it. `zoom_at` (s) shows "
                    "60 m around the pair at that moment.",
     "schema": {"type": "object", "properties": {"marks": {"type": "array", "items": {"type": "number"}},
                                                 "zoom_at": {"type": "number"}}},
     # matplotlib runs in the caller's thread on purpose: pyplot's global state is not thread-safe
     "call": lambda scene, a, fig_dir: ("image", T.render(scene, a.get("marks"), a.get("zoom_at"), out_dir=fig_dir))},
]


# --- prompts ---------------------------------------------------------------------------------

def rules_text(skill_dir, profile):
    skill = Path(skill_dir)
    rubric = (skill / "references" / "rubric.md").read_text(encoding="utf-8")
    body = (skill / "SKILL.md").read_text(encoding="utf-8")
    start = body.index("## Rules of judgement")
    judgement = body[start:body.index("\n## ", start + 5)]
    return "%s\n\n%s\n\n## Jurisdiction profile in force\n\n```json\n%s\n```" % (
        rubric, judgement, json.dumps(profile, indent=1, ensure_ascii=False))


REVIEWER = """You review driving logs from the Waymo Open Motion Dataset and decide whether a driver's
behaviour toward one other road user was aggressive. You are an expert in traffic law and in how
people actually drive, and you judge for yourself: two automatic screens have looked at each scene
first, and their rows are evidence, not verdicts. They have known blind spots (described below);
people's earlier labels are not available to you and would not be ground truth anyway.

Each scene has a pair: the "first" and the "second" agent (the overview says which is the SDC).
Judge the first toward the second and the second toward the first, ignoring how either treats third
parties. Check every claim against the numbers and the figure: who actually moved, who arrived
first, whether the other paid for it (was made to stop again, braked hard, was cut close, was left a
headway it had no say in). A third road user may explain a brake or a lane change; look before you
blame. The first message already holds the scene overview, the pair's time series every 0.5 s
and, where the model takes images, the default figure: that settles most scenes. Call a tool only
for what they cannot show (a third road user, a finer time window, a zoomed figure).

Answer in the JSON schema. Each reason is one English sentence naming the deciding rule with its
numbers (times, headways, braking, speeds). Use "low" confidence when the data cannot settle it
(clip starts too late, track glitches, map missing), and set needs_second_look when you are unsure
or overrule the screens.

# The rules

{rules}
"""

CRITIC = """You are the second reviewer for driving-aggression verdicts on Waymo Open Motion Dataset
scenes. A first reviewer has judged a scene and disagreed with the automatic screens (or was unsure).
Look at the scene yourself (the first message holds the overview, the time series and, where the
model takes images, the figure; the tools show more), then make the strongest honest case against the first
reviewer's verdict: wrong actor, misread geometry, a third party that explains it, a lawful
manoeuvre called aggressive, or a real cost to the other road user that was waved away. Then give
the final verdict in the JSON schema (upheld = the first reviewer's verdict stands on both sides).
Do not split the difference: if the evidence supports one reading, give it; if it supports neither,
say so with low confidence.

# The rules

{rules}
"""

AUDITOR = """You audit an automatic screen for aggressive driving against the verdicts of reviewers
who looked at the scenes. You may read the run folder (labels, disagreements, the reviewers' JSON)
and the skill's scripts and references. Propose changes to the screens' rules, thresholds or code
that would remove whole classes of disagreement, each with: the scenes it explains, the change
(file, function, what to do), what it might break, and whether it risks fitting the reviewers rather
than the law or common sense. Do not propose changes driven by one scene. Write markdown. Nothing
you propose is applied without a person's approval; do not edit files.
"""


def brief(name, figure=False):
    """What a scene agent is given up front, so it need not spend turns asking for it: the overview
    (the screens' rows and the triage reason included) and the time series every 0.5 s. ``figure``:
    True (a figure is attached), False (the agent has the tools) or None (batch: neither)."""
    note = {True: "\n\nThe attached image is render_scene with its default marks.",
            False: "\n\nNo figure yet: call render_scene if the geometry matters.", None: ""}[figure]
    return "# scene_overview\n\n%s\n\n# pair_timeseries, 0-9 s every 0.5 s\n\n%s%s" % (
        T.overview(name), T.timeseries(name, 0.0, 9.0, 0.5), note)


def review_prompt(name, figure=False):
    return "Judge scene %s.\n\n%s\n\nGive your verdict." % (name, brief(name, figure))


def critic_prompt(name, first_verdict, figure=False):
    return ("Scene %s. The first reviewer's verdict:\n```json\n%s\n```\n\n%s\n\nCheck it and give the final "
            "verdict." % (name, json.dumps(first_verdict, indent=1, ensure_ascii=False), brief(name, figure)))


# --- batches: several scenes in one request, text only, no tools -------------------------------

_TOOLS_NOTE = """The first message already holds the scene overview, the pair's time series every 0.5 s
and, where the model takes images, the default figure: that settles most scenes. Call a tool only
for what they cannot show (a third road user, a finer time window, a zoomed figure)."""
_BATCH_NOTE = """The message holds several scenes, each with its overview and the pair's time series
every 0.5 s; there are no tools and no figures. Judge each scene on its own. Where the numbers
cannot settle a scene, say so with low confidence rather than guess."""
assert _TOOLS_NOTE in REVIEWER
REVIEWER_BATCH = REVIEWER.replace(_TOOLS_NOTE, _BATCH_NOTE)
_CRITIC_TOOLS = """(the first message holds the overview, the time series and, where the
model takes images, the figure; the tools show more)"""
assert _CRITIC_TOOLS in CRITIC
CRITIC_BATCH = CRITIC.replace(_CRITIC_TOOLS, "(each scene's overview and time series are given; there are no tools)")


def batch_schema(verdict):
    one = dict(verdict, properties=dict(verdict["properties"], scene={"type": "string"}),
               required=["scene"] + verdict["required"])
    return {"type": "object", "properties": {"verdicts": {"type": "array", "items": one}},
            "required": ["verdicts"], "additionalProperties": False}


def batch_review_prompt(names):
    return ("Judge each of these %d scenes and return one verdict per scene, with its scene id, in the order given."
            "\n\n%s" % (len(names), "\n\n".join("## Scene %s\n\n%s" % (n, brief(n, None)) for n in names)))


def batch_critic_prompt(items):
    """``items``: (scene, first reviewer's verdict) pairs."""
    return ("Check the first reviewer's verdict on each of these %d scenes and return one final verdict per scene, "
            "with its scene id, in the order given.\n\n%s" % (len(items), "\n\n".join(
                "## Scene %s\n\nThe first reviewer's verdict:\n```json\n%s\n```\n\n%s" % (
                    n, json.dumps(v, indent=1, ensure_ascii=False), brief(n, None)) for n, v in items)))


def audit_prompt(run_dir):
    return ("The run folder is %s (labels.csv, disagreements.csv, reviews/*.json); the skill is the current folder "
            "(scripts/screen_crossing.py, scripts/screen_same_direction.py, references/rubric.md). Write the rule "
            "proposals." % run_dir)


def inside(roots, target):
    """Whether ``target`` (relative paths are taken from roots[0]) lies inside one of ``roots``."""
    p = Path(target)
    p = (roots[0] / p if not p.is_absolute() else p).resolve()
    return any(p == r or r in p.parents for r in roots)
