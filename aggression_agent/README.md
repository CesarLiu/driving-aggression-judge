# aggression_agent

Labels aggressive driving in WOMD-style scenes (ScenarioNet pickles or catk's `womd_cache`) with
the `judge-driving-aggression` skill's screens plus reviewing agents. The agents run on Claude
through the Claude Agent SDK (default), or on a local model behind an OpenAI-compatible server such
as vLLM or Ollama (`--backend openai`; see "Local models").
The screens do the arithmetic on every scene. Agents look only at scenes the screens flagged,
doubted or cannot see, and judge those with the numbers and a figure, the way a person reviewing
the screen's output would.

```
scenes ──► screens (no model) ──► triage ──► reviewer agent ──► critic agent ──► labels.csv
           crossing, same-dir.     │          (every triaged      (only where the     disagreements.csv
                                   │           scene)              reviewer overruled  report.md
                                   │                               the screens or was  reviews/*.json
                                   └─ the rest keep the            unsure)             figures/*.png
                                      screens' verdict
                                                         runs ──► rule auditor ──► rule_proposals.md
                                                                  (proposals only, nothing applied)
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or later. The SDK bundles Claude Code. It authenticates like Claude Code: with an
existing `claude` login, or with `ANTHROPIC_API_KEY` (from https://console.anthropic.com/) set in
the environment; `.env.example` names it, but nothing loads a `.env` file automatically. The local
backend (`--backend openai`) needs only the standard library. The screens need only NumPy (and
matplotlib for the figures): the scene and lane code they use is copied into the skill's
`scripts/responsibility/`. The skill is found at `../.claude/skills/judge-driving-aggression`
(this repository's layout), or wherever `--skill-dir` / `$AGGRESSION_SKILL_DIR` points.

## Run

```bash
# CAT's 500 scenes (ScenarioNet pickles); both screens run first
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500

# reuse screen output already made
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500 \
  --crossing-csv screen_crossing.csv --same-direction-csv screen_same_direction.csv

# catk's cached WOMD (validation split)
python main.py run --scenes /data/catk/womd_cache/validation --out runs/catk_val

python main.py run ... --screen-only        # stop after triage: see what would be reviewed, no cost
python main.py run ... --only 237,293       # a few scenes
python main.py propose --run runs/cat500    # rule proposals from the run's disagreements
```

Runs resume: screen CSVs and finished reviews in the run folder are reused. Useful options include
`--concurrency` (default 3), `--critic disagree|all|none`, `--reviewer-model` (default
`claude-haiku-5-5`), `--critic-model` (default `claude-sonnet-5-5`), `--budget` (USD cap per reviewer
run; the critic gets 1.5x) and `--profile` (jurisdiction).

## Local models (vLLM, Ollama)

A model you serve yourself can do the judging in three ways. The first two reuse a setup that
already works in Claude Code: `ANTHROPIC_BASE_URL` pointing at the server, plus
`ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` if it wants one. The third uses the server's
OpenAI-compatible API. To find the served model name:

```bash
curl http://<host>:<port>/v1/models
```

Whichever way you choose, first run the model on scenes already reviewed: 237, 250, 293 and 342,
and the full CAT run's `runs/cat500/labels.csv` from the Claude backend. Compare the verdicts before
trusting its labels. None of the three has been run with a real local model yet.

### 1. The skill in Claude Code, one scene at a time

Start Claude Code with the local model and open this repository. The skill in
`.claude/skills/judge-driving-aggression/` loads by itself; ask, for example, "use
judge-driving-aggression to judge whether the SDC in scene 237 of /data/cat/raw_scenes_500 is
aggressive". The model follows `SKILL.md`: it runs the screens, reads the tracks and writes the
labels.

Nothing in this folder is involved. This suits a few scenes, and its quality is the model's own
skill at following `SKILL.md` with Claude Code's tools.

### 2. This pipeline with the Claude backend, pointed at the server

The Claude Agent SDK runs Claude Code and inherits the shell's environment, so the same variables
send the agents to your server. Name the served model for both agents: the critic's default is a
Claude model.

```bash
# in a shell where ANTHROPIC_BASE_URL (and the token, if any) point at the server, as for Claude Code
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/vllm_check --only 237,250,293,342 \
  --reviewer-model <served-model-name> --critic-model <served-model-name> --concurrency 1
```

The verdicts come back through Claude Code's structured-output tool, and the figure through an
image block in a tool result. The model must call that tool reliably. For the figure to reach it,
the model must accept images and the server's Anthropic-compatible endpoint must pass images inside
`tool_result`. A review that failed shows `subtype` other than `success` and no `final` in
`reviews/<scene>.json`; then use way 3. The cost in the report is priced as Claude and means
nothing here. `--no-images` does not apply to this backend.

### 3. This pipeline with the OpenAI-compatible backend

```bash
# vLLM: serve with tool calling enabled (the parser must match the model family)
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500_qwen --backend openai \
  --base-url http://localhost:8000/v1 --reviewer-model Qwen/Qwen2.5-VL-72B-Instruct

# Ollama
python main.py run ... --backend openai --base-url http://localhost:11434/v1 --reviewer-model qwen2.5vl:32b

python main.py run ... --backend openai --no-images ...   # a text-only model: no figure tool
python main.py propose --run runs/cat500_qwen --backend openai --base-url ... --model <name>
```

A vLLM server started without `--enable-auto-tool-choice --tool-call-parser ...` cannot do function
calling and has to be restarted with them. `local_agents.py` runs the same agents, prompts, scene
tools and records (`spec.py`) through the chat-completions API with function calling. It is written
for local models' habits:

- **Verdict.** It is handed in through a `submit_verdict` function whose parameters are the verdict
  schema. An invalid verdict is sent back with its errors. A verdict written as plain JSON is also
  accepted. On the last turn the call is forced (`tool_choice`).
- **Figures.** A figure goes in a user message after the tool results, because the chat API has no
  images in tool results. Only the newest `--max-images` (default 2) stay in the conversation.
- **Model output.** `<think>` blocks are stripped. Nested arguments sent as JSON strings, and
  `"true"`/`"false"`, are converted.
- **Critic model.** `--critic-model` defaults to the reviewer model. There is no cost cap (only
  `--max-turns`), and token counts go to `usage` in `reviews/*.json`.
- **Auditor.** `propose` gets its own read-only `list_files` / `read_file` / `grep`, limited to the
  skill and the run folder.

The model must call functions reliably over many turns. With figures it must also accept images (a
Qwen2.5-VL / Qwen3-VL class model). Small models often lose the thread in multi-turn tool use, and
none has been validated on these scenes. Run a new model first on scenes already reviewed (237,
250, 293, 342, and the full CAT run's `runs/cat500`), and compare before trusting its labels.

The loop was tested against a scripted mock server, not a real local model. The mock covered tool
calls, several calls in one turn, the figure message, rejection and correction of an invalid
verdict, a plain-JSON verdict inside `<think>`, the forced last turn, `--no-images`, figure pruning
and the auditor's folder limit.

## What goes to review (triage, no model)

- A right-of-way violation found by the crossing screen, whether or not it cost the other anything.
- A same-direction candidate: cut-in, tailgating or close pass.
- A screen note marking a borderline or doubtful case: `borderline_cut_in_thw`,
  `cut_in_pulling_away`, or the sanity checks `implausible_lateral_move` and `boxes_touch`.
- A lane join that the screen credited to neither car. This is the screen's lane-branch blind spot,
  where a car follows a lane link across the road. All seven known cases in CAT (15, 69, 156, 293,
  337, 381, 483) carry this signature.
- Two cars that start side by side and end up one behind the other, with the screen seeing no lane
  change.

On CAT's 500 scenes, 154 go to review. Every other scene keeps the screens' verdict, not
aggressive, with `source = screen`.

## The agents

- **Reviewer** (one per scene) has four in-process tools (`scene_tools.py`):
  - `scene_overview`: the pair, speeds, stops, braking, turn, gaps, the screens' rows and the
    triage reason;
  - `pair_timeseries`: positions, speeds, braking held over 1 s, box gap, relative position,
    follower, headway and TTC;
  - `nearby_road_users`: third parties that could explain a brake or a lane change;
  - `render_scene`: map, tracks and boxes, plus speed and gap plots, returned to the agent as an
    image and saved to `figures/`.

  It judges both directions and answers in a JSON schema (aggressive, kind, a reason with numbers,
  confidence; agreement with the screens; whether it wants a second look).
- **Critic** looks again where the reviewer overruled the screens, gave low confidence or asked for
  a second look. It is told to make the strongest honest case against the reviewer, then give the
  final verdict. Its verdict is final (`source = critic`).
- **Rule auditor** (`propose`) has read-only file tools on the skill and the run folder. It writes
  proposals to `rule_proposals.md`: the change (file, function), the scenes it explains, what it
  might break, and whether it risks fitting reviewers rather than the law. Nothing is applied
  without a person.

Design choices:

- **The agents are blind.** They have no human labels, no built-in tools (no files, no shell) and
  no user or project settings (`setting_sources=[]`, so no CLAUDE.md). Auto-memory is off, and no
  MCP servers other than the scene tools load (`strict_mcp_config`). Each reviewer's tools are bound
  to its own scene. Each `reviews/*.json` records the tool list the agent started with, so this can
  be checked: it is the four scene tools plus `StructuredOutput`. The auditor's file tools are held
  to the skill and run folders by a PreToolUse hook.
- **Failures stay local.** An agent run that fails is recorded with its error and retried on resume;
  it does not cancel the other scenes.
- **The rules come from the skill.** The rules they judge by are read at start-up from the skill's
  `references/rubric.md`, the "Rules of judgement" in `SKILL.md`, and the jurisdiction profile. The
  framework cannot drift from the skill.
- **The screens' rows are evidence, not verdicts.**

## Outputs

`labels.csv` has one row per screened scene. The `sdc_*` columns describe the pair's first agent
and the `ooi_*` columns its second, as in the screens. The columns are: aggressive, kind, reason,
confidence; `source` (screen, reviewer, critic or not_reviewed); `screen_sdc` and `screen_ooi` (what
the screens said); and `triage`.

The crossing screen's right-of-way facts follow, copied unchanged, so that a violation which cost
the other nothing is still visible:
- `row_case`: the rule that decided;
- `row_holder`: who had priority (`sdc`, `ooi` or `none`);
- `row_went_first` and `row_pet_s`: who reached the conflict point first, and how many seconds
  apart;
- `row_affected`: whether the other vehicle paid for a violation;
- `sdc_row_violation` and `ooi_row_violation`;
- `sdc_red_entry` and `ooi_red_entry`: how each vehicle entered on red under the profile, for
  example `lawful_turn_on_red`, `ran_red` or `turn_on_red_needs_sign`. These two are empty when
  the vehicle did not enter on red, or when the crossing CSV predates them.

`disagreements.csv` lists the reviewed scenes whose final verdict differs from the screens.
`report.md` summarises them. `reviews/<scene>.json` holds every agent's verdict, tool calls, turns
and cost.

## Measured so far (2026-10-28)

These runs used the earlier defaults: `claude-sonnet-5-5` as reviewer and `claude-opus-5-5` as
critic. Since 2026-10-30 the defaults are `claude-haiku-5-5` and `claude-sonnet-5-5`; their
verdicts and costs have not been measured yet.

The smoke test used CAT scenes 23, 237, 250, 293 and 342, plus catk 48824446f21b3eb0 and
975a3faf7536fc35.

- **Cost.** A reviewer took 5 turns and about $0.05; a critic took about $0.14. The first call of a
  run costs more (about $0.2) until the prompt cache is warm.
- **Verdicts.**
  - The blind-spot cut-in in 293 was found: the SDC moved in ahead of the OOI, slightly slower than
    it, at 0.66-0.72 s headway.
  - The screen-only OOI cut-in in 250 was confirmed: 0.37-0.43 s headway.
  - In 342 the SDC cut-in was kept and the OOI "tailgate" was dropped, because it fell within 3 s of
    being cut in on.
  - In 237 the screen's SDC cut-in was overruled. The SDC was 2.1 m/s faster at 0.72 s and the
    follower's 1.1 m/s² was speed jitter, so the rubric's pull-away exemption applies.
- **The auditor's first proposal on that run.** The screen cancels the pull-away exemption at
  1.0 m/s² of braking (hard-coded in `directed`), while the rubric counts only 2 m/s² as a cost.
  This explains 237. Not applied.

### Full CAT run (500 scenes, Claude backend, 2026-10-28)

The screens' output was reused: crossing `screen_law_us_v4`, same-direction `v8`.

- **Coverage.** 154 scenes were sent for review and all 154 got a verdict. The critic was final in
  51, and no agent run failed. Cost was $9.47 at API prices; the run itself used a Claude Code
  subscription login, at `--concurrency 4`.
- **Final labels.**
  - First agent (the SDC in every CAT pair): cut-in 6.
  - Second (OOI): cut-in 28, tailgating 6.
  - No crossing conflict was aggressive, which agrees with the crossing screen's 0 candidates.
- **Against the screens (first agent / second).**
  - The screens had cut-in 6 / 32 and tailgating 2 / 8.
  - The final verdict differs in 19 scenes:
    - 14 screen findings were dropped (cut-in 8, tailgating 6);
    - 6 were added that the screens missed (cut-in 4, tailgating 2).
- **What the dropped findings were.**
  - **Faster merger, no forced braking (1, 51, 75, 237).** The merging car was 2 m/s or more faster
    and left 0.5-0.8 s; the rubric's exemption applies. The screen cancels it once the follower
    brakes 1 m/s².
  - **Headway taken at the first sideways overlap (183, 294, 449).** The screen reads the headway
    when the boxes first overlap by a few centimetres, while the merging car is still mostly in
    its own lane. A moment later the headway is above 0.5 s and rising.
  - **Borderline headway with a queue ahead (60).** The headway was 0.80-0.87 s and the queue
    ahead explains the follower's braking.
  - **Tailgating that was not.**
    - 79: a launch from standstill with the gap opening.
    - 235: the approach to a pass, below 0.9 s for 1.2 s.
    - 337: the SDC was cut in on, not tailgating.
    - 342: inside the 3 s after being cut in on.
    - 411: a turning car 6-7 m to the side.
    - 495: a gap of 21 m or more, a lane-match artefact.
- **What the added findings were.**
  - **Cut-ins in the lane-branch blind spot** (219, 293, 496; 337's OOI merging from an ending
    lane). The headway was 0.40-0.66 s.
  - **Tailgating from half a lane over** (290, 437). In 290 the OOI moved in 3.3-3.5 m behind the SDC
    at 14.5-15 m/s, 0.22-0.25 s.

These are agent verdicts, checked by a person only for 237, 250, 293 and 342. Read
`reviews/*.json` and the figures before trusting a label. The two systematic screen issues (the
1 m/s² exemption cut-off and headway at the first overlap) are candidates for a screen change. That
change should be checked against these scenes, not tuned to them.
