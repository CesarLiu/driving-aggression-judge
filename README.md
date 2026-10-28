# driving-aggression-judge

Decides whether a driver in a WOMD-style scene behaved aggressively toward one other road user.

- **Conflicts at intersections** (crossing, turning) are judged by the right of way under a chosen
  traffic code.
- **Same-direction conflicts** (cut-ins, tailgating, close passes) are judged by physical rules.

"Aggressive" means: X's own action shifted risk or effort onto Y beyond what the situation required.
Breaking the right of way counts only if it cost the other vehicle something.

Input is a folder of scene pickles: ScenarioNet / MetaDrive scenes (e.g. CAT's `raw_scenes_500`) or
catk's cached WOMD (`womd_cache/<split>`). No data is included.

## Layout

```
.claude/skills/judge-driving-aggression/   a Claude Code skill: the method, usable by hand or by an agent
  SKILL.md                                 workflow and rules of judgement
  scripts/screen_crossing.py               right-of-way screen (jurisdiction profiles)
  scripts/screen_same_direction.py         cut-in / tailgating / close-pass screen
  scripts/scenes.py                        loads both scene formats, picks the pair to judge
  scripts/jurisdiction.py                  profile engine (self-test: python scripts/jurisdiction.py)
  scripts/responsibility/                  scene, lane-graph and right-of-way code from the CAT fork
  jurisdictions/                           us-california (validated by review), drafts for others
  references/                              rubric, profile schema, what was validated and its limits
  assets/, evals/                          a person's labels on CAT's 500 scenes; skill trigger evals
aggression_agent/                          runs the method unattended over a dataset with agents
  main.py                                  CLI: run, propose
  pipeline.py                              screens → triage → reviewer → critic → labels
  agents.py / local_agents.py              Claude (Claude Agent SDK) / OpenAI-compatible (vLLM, Ollama)
  spec.py, scene_tools.py                  shared prompts and schemas; the agents' scene tools
```

## Use

```bash
pip install -r aggression_agent/requirements.txt        # numpy, matplotlib, claude-agent-sdk

# the screens alone (NumPy only)
python .claude/skills/judge-driving-aggression/scripts/screen_crossing.py --scenes /data/cat/raw_scenes_500 --profile us-california --out screen_crossing.csv
python .claude/skills/judge-driving-aggression/scripts/screen_same_direction.py --scenes /data/cat/raw_scenes_500 --out screen_same_direction.csv

# screens + reviewing agents over a dataset
cd aggression_agent
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500                       # Claude
python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500_local --backend openai \
  --base-url http://localhost:8000/v1 --reviewer-model <served model>                        # vLLM / Ollama
```

Opened in Claude Code, the repository's skill loads by itself: ask for example "is the SDC
aggressive in these scenes". `aggression_agent/README.md` describes the triage, the agents, the
outputs and what was measured. `references/validation-us-california.md` describes what the
California rules were checked against.

## Dependencies

The code does not depend on the CAT repository or on PyTorch. The few modules of the CAT fork's
`responsibility` package that the screens need are copied into the skill's `scripts/responsibility/`
(source commit and changes are in its `__init__.py`; CAT's MIT licence is in `LICENSE-CAT`). The
Claude backend needs `claude-agent-sdk`, which bundles Claude Code and uses a Claude Code login or
`ANTHROPIC_API_KEY`. The local backend needs only the standard library and a server with the
OpenAI chat API and function calling.
