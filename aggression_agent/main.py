"""Label aggressive driving in WOMD-style scenes with screens plus reviewing agents.

    python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500
    python main.py run --scenes /data/catk/womd_cache/validation --out runs/catk_val --concurrency 4
    python main.py propose --run runs/cat500          # rule proposals from a finished run's disagreements

    # a local model behind an OpenAI-compatible server (vLLM, Ollama)
    python main.py run --scenes /data/cat/raw_scenes_500 --out runs/cat500_qwen --backend openai \\
        --base-url http://localhost:8000/v1 --reviewer-model Qwen/Qwen2.5-VL-72B-Instruct

See README.md for what each stage does and what it costs.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SKILL = HERE.parent / ".claude" / "skills" / "judge-driving-aggression"


def add_backend(parser):
    g = parser.add_argument_group("backend")
    g.add_argument("--backend", default="claude", choices=("claude", "openai"),
                   help="claude: Claude Agent SDK (Claude Code login or ANTHROPIC_API_KEY); openai: any "
                        "OpenAI-compatible server with function calling, e.g. local vLLM or Ollama")
    g.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                   help="openai backend: server URL ending in /v1 (vLLM :8000/v1, Ollama :11434/v1)")
    g.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                   help="openai backend: bearer token, if the server wants one ($OPENAI_API_KEY)")
    g.add_argument("--no-images", action="store_true",
                   help="openai backend: no figure tool, for text-only models")
    g.add_argument("--max-images", type=int, default=2, help="openai backend: figures kept in the conversation")
    g.add_argument("--request-timeout", type=float, default=600.0, help="openai backend: seconds per request")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="screen, triage, review, write labels")
    r.add_argument("--scenes", required=True, help="folder of scene pickles (ScenarioNet or catk womd_cache)")
    r.add_argument("--out", required=True, help="run folder; an existing one is resumed")
    r.add_argument("--profile", default="us-california", help="jurisdiction profile of the skill")
    r.add_argument("--pairs", default="auto", choices=("auto", "ooi", "sdc-nearest"))
    r.add_argument("--only", help="comma-separated scene names (file stems) to label")
    r.add_argument("--crossing-csv", help="reuse this crossing screen output instead of screening again")
    r.add_argument("--same-direction-csv", help="reuse this same-direction screen output")
    r.add_argument("--screen-only", action="store_true", help="stop after triage (no agents, no cost)")
    r.add_argument("--no-blind-spot-scan", action="store_true",
                   help="skip the scan for side-by-side cars that end up in one lane (loads every scene)")
    r.add_argument("--max-review", type=int, help="review at most this many triaged scenes")
    r.add_argument("--concurrency", type=int, default=3, help="scenes reviewed at once")
    r.add_argument("--critic", default="disagree", choices=("disagree", "all", "none"),
                   help="when the critic looks again: when the reviewer overruled the screens or was unsure (default)")
    r.add_argument("--reviewer-model", help="default claude-sonnet-5-5 (claude backend); required with openai")
    r.add_argument("--critic-model", help="default claude-opus-5-5 (claude backend); default the reviewer model with openai")
    r.add_argument("--budget", type=float, default=1.0, help="USD cap per reviewer run (critic: 1.5x; claude backend only)")
    add_backend(r)
    r.add_argument("--max-turns", type=int, default=24)
    r.add_argument("--skill-dir", default=os.environ.get("AGGRESSION_SKILL_DIR", str(DEFAULT_SKILL)))

    p = sub.add_parser("propose", help="rule proposals from a run's disagreements (nothing is applied)")
    p.add_argument("--run", required=True)
    p.add_argument("--model", help="default claude-opus-5-5 (claude backend); required with openai")
    p.add_argument("--budget", type=float, default=3.0)
    p.add_argument("--skill-dir", default=os.environ.get("AGGRESSION_SKILL_DIR", str(DEFAULT_SKILL)))
    add_backend(p)

    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    claude = args.backend == "claude"
    if args.cmd == "run":
        if claude:
            args.reviewer_model = args.reviewer_model or "claude-sonnet-5-5"
            args.critic_model = args.critic_model or "claude-opus-5-5"
        elif not args.reviewer_model:
            ap.error("--backend openai needs --reviewer-model (the model name the server serves)")
        else:
            args.critic_model = args.critic_model or args.reviewer_model
        import pipeline

        asyncio.run(pipeline.run_pipeline(args))
    else:
        if not claude and not args.model:
            ap.error("--backend openai needs --model")
        import pipeline

        agents = pipeline.load_backend(args)
        out = asyncio.run(agents.audit(args.run, args.skill_dir, args.model or "claude-opus-5-5", args.budget))
        path = Path(args.run) / "rule_proposals.md"
        path.write_text(out["markdown"] or "(no output: %s)" % out["subtype"], encoding="utf-8")
        print("wrote %s ($%.2f, %s)" % (path, out["cost_usd"], out["subtype"]))


if __name__ == "__main__":
    main()
