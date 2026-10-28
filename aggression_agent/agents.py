"""Claude backend: the agents run with the Claude Agent SDK (``--backend claude``, the default).

* **Reviewer**: judges one scene, both directions (first toward second, second toward first), from
  the screens' rows, the numbers and a figure. It has only the scene tools: no files, no shell, no
  settings or memory of the user, and never any human labels.
* **Critic**: looks at the same scene again where the reviewer disagreed with the screens or was
  unsure, tries to build the strongest case against the reviewer's verdict, and gives the final one.
* **Rule auditor**: reads a finished run's disagreements and the screens' code and proposes changes
  to rules or thresholds as text. Nothing it proposes is applied.

Schemas, prompts and tool specifications are shared with the other backend (``spec.py``). The SDK
authenticates like Claude Code: an existing ``claude`` login, or ``ANTHROPIC_API_KEY``.
"""

import base64
import json
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage, SystemMessage,
                              TextBlock, ToolUseBlock, create_sdk_mcp_server, query, tool)

import spec
from spec import rules_text  # noqa: F401  (the pipeline reads the rules through the backend)


# --- tools -----------------------------------------------------------------------------------

def _sdk_tool(t, scene, fig_dir):
    @tool(t["name"], t["description"], t["schema"])
    async def run(args):
        try:
            kind, out = t["call"](scene, args, fig_dir)
        except Exception as exc:  # the agent sees the error and can correct its call
            return {"content": [{"type": "text", "text": "%s: %s" % (type(exc).__name__, exc)}], "is_error": True}
        if kind == "image":
            return {"content": [{"type": "image", "data": base64.b64encode(out).decode(), "mimeType": "image/png"}]}
        return {"content": [{"type": "text", "text": out}]}
    return run


def make_server(fig_dir, scene):
    """The scene tools, bound to one scene: an agent can only look at the scene it judges."""
    tools = [_sdk_tool(t, scene, fig_dir) for t in spec.SCENE_TOOLS]
    return create_sdk_mcp_server("scene", tools=tools), ["mcp__scene__" + t["name"] for t in spec.SCENE_TOOLS]


# --- running an agent ------------------------------------------------------------------------

async def _run(prompt, options):
    calls, texts, result, init = [], [], None, {}
    async for m in query(prompt=prompt, options=options):
        if isinstance(m, SystemMessage) and m.subtype == "init":
            init = {k: m.data.get(k) for k in ("tools", "mcp_servers", "skills", "plugins", "model")}
        elif isinstance(m, AssistantMessage):
            for block in m.content:
                if isinstance(block, ToolUseBlock):
                    calls.append({"tool": block.name.split("__")[-1], "input": block.input})
                elif isinstance(block, TextBlock) and block.text.strip():
                    texts.append(block.text)
        elif isinstance(m, ResultMessage):
            result = m
    out = None
    if result is not None and result.structured_output is not None:
        out = result.structured_output
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                out = None
    return {"verdict": out, "subtype": getattr(result, "subtype", "no_result"),
            "is_error": getattr(result, "is_error", None), "errors": getattr(result, "errors", None),
            "terminal_reason": getattr(result, "terminal_reason", None), "init": init,
            "cost_usd": getattr(result, "total_cost_usd", None) or 0.0,
            "turns": getattr(result, "num_turns", None), "tool_calls": calls, "text": texts,
            "result_text": getattr(result, "result", None)}


# The agents must not see the user's memory (it holds notes about earlier labels) or any connector.
ISOLATED_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


def _options(scene, system, model, fig_dir, schema, budget, max_turns):
    server, names = make_server(fig_dir, scene)
    return ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        mcp_servers={"scene": server},
        strict_mcp_config=True,  # only the scene server: no account connectors or .mcp.json servers
        env=ISOLATED_ENV,
        tools=[],  # no built-in tools: the scene tools only
        allowed_tools=names,
        permission_mode="dontAsk",
        setting_sources=[],  # no user or project settings, CLAUDE.md or memory: a clean, blind reviewer
        max_turns=max_turns,
        max_budget_usd=budget,
        output_format={"type": "json_schema", "schema": schema},
    )


async def review(name, rules, model, fig_dir, budget=1.0, max_turns=24):
    return await _run(spec.review_prompt(name),
                      _options(name, spec.REVIEWER.format(rules=rules), model, fig_dir, spec.VERDICT, budget, max_turns))


async def criticise(name, first_verdict, rules, model, fig_dir, budget=1.5, max_turns=24):
    return await _run(spec.critic_prompt(name, first_verdict),
                      _options(name, spec.CRITIC.format(rules=rules), model, fig_dir, spec.CRITIC_VERDICT, budget,
                               max_turns))


def _inside(roots):
    """A PreToolUse hook that denies Read/Grep/Glob outside ``roots``: with permission_mode dontAsk the
    allow list alone would let the auditor read any file on the machine."""
    async def check(data, tool_use_id, context):
        args = data.get("tool_input", {})
        target = args.get("file_path") or args.get("path")
        if target is None or spec.inside(roots, target):
            return {}  # Grep/Glob without a path search the working directory, the skill
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": "only the skill and the run folder may be read"}}
    return check


async def audit(run_dir, skill_dir, model, budget=3.0):
    run_dir, skill_dir = Path(run_dir).resolve(), Path(skill_dir).resolve()
    options = ClaudeAgentOptions(
        system_prompt=spec.AUDITOR,
        model=model,
        tools=["Read", "Grep", "Glob"],
        allowed_tools=["Read", "Grep", "Glob"],
        permission_mode="dontAsk",
        setting_sources=[],
        strict_mcp_config=True,
        env=ISOLATED_ENV,
        hooks={"PreToolUse": [HookMatcher(matcher="Read|Grep|Glob", hooks=[_inside([skill_dir, run_dir])])]},
        cwd=str(skill_dir),
        add_dirs=[str(run_dir)],
        max_turns=40,
        max_budget_usd=budget,
    )
    out = await _run(spec.audit_prompt(run_dir), options)
    out["markdown"] = out["result_text"] or "\n\n".join(out["text"])
    return out
