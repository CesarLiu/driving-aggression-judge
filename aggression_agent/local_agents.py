"""OpenAI-compatible backend: the agents run against any server with the OpenAI chat-completions API
and function calling (``--backend openai``), e.g. a local vLLM (``vllm serve <model>
--enable-auto-tool-choice --tool-call-parser <parser>``, base URL http://host:8000/v1) or Ollama
(base URL http://host:11434/v1).

Same agents, prompts, scene tools and records as the Claude backend (``spec.py``), with a tool loop
written here because these servers only answer one request at a time:

* the verdict is handed in through a ``submit_verdict`` function whose parameters are the verdict
  schema, checked with ``spec.validate``; an invalid one is sent back with the errors. A verdict
  written as plain JSON text is accepted too. On the last turn the call is forced (``tool_choice``).
* the chat API cannot put an image in a tool result, so a figure goes in a user message right after
  the tool results. Only the newest ``max_images`` figures stay in the conversation (small context
  windows); ``--no-images`` drops the figure tool for text-only models.
* reasoning wrapped in ``<think>...</think>`` (Qwen3, DeepSeek-R1 style) is removed before parsing.

The model must call functions reliably and, with figures, accept images (e.g. a Qwen2.5-VL / Qwen3-VL
class model). No cost is computed; token counts are recorded instead.
"""

import asyncio
import base64
import fnmatch
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import spec
from spec import rules_text  # noqa: F401  (the pipeline reads the rules through the backend)


@dataclass
class Config:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    images: bool = True
    max_images: int = 2
    temperature: float = 0.0
    timeout: float = 600.0
    retries: int = 2


CFG = Config()


def configure(**kw):
    for k, v in kw.items():
        if v is not None:
            setattr(CFG, k, v)


# --- HTTP ------------------------------------------------------------------------------------

def _post(body):
    req = urllib.request.Request(CFG.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + CFG.api_key})
    last = None
    for attempt in range(CFG.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=CFG.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            last = RuntimeError("HTTP %s from %s: %s" % (exc.code, CFG.base_url, detail))
            if exc.code < 500:
                break  # a bad request will not get better by asking again
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = RuntimeError("cannot reach %s: %s" % (CFG.base_url, exc))
    raise last


async def _chat(model, messages, tools=None, tool_choice=None):
    body = {"model": model, "messages": messages, "temperature": CFG.temperature}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    return await asyncio.to_thread(_post, body)


def _function(name, description, schema):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


# --- parsing what a local model sends back ---------------------------------------------------

THINK = re.compile(r"<think>.*?</think>", re.S)


def _strip(text):
    return THINK.sub("", text or "").strip()


def _arguments(raw):
    if isinstance(raw, dict):
        return raw
    return json.loads(raw or "{}")


def _json_in(text):
    """The JSON object in a reply: a ```json block, else the outermost braces."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    body = m.group(1) if m else text[text.find("{"):text.rfind("}") + 1]
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError:
        return None


def _coerce(value, schema):
    """Undo what small models often do to nested arguments: objects sent as JSON strings, booleans as
    "true"/"false"."""
    kind = schema.get("type")
    if kind == "object" and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if kind == "object" and isinstance(value, dict):
        return {k: _coerce(v, schema.get("properties", {}).get(k, {})) for k, v in value.items()}
    if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _trim_images(messages, keep):
    seen = 0
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for k, part in enumerate(m["content"]):
                if part.get("type") == "image_url":
                    seen += 1
                    if seen > keep:
                        m["content"][k] = {"type": "text", "text": "[an earlier figure, removed to save context]"}


def _record(model, tools, verdict, subtype, turns, calls, texts, usage, errors=None, final_text=None):
    return {"verdict": verdict, "subtype": subtype, "is_error": verdict is None and subtype != "success",
            "errors": errors, "terminal_reason": subtype,
            "init": {"backend": "openai-compatible", "base_url": CFG.base_url, "model": model, "tools": tools},
            "cost_usd": 0.0, "usage": usage, "turns": turns, "tool_calls": calls, "text": texts,
            "result_text": final_text}


# --- the scene agents ------------------------------------------------------------------------

async def _scene_agent(scene, system, prompt, schema, model, fig_dir, max_turns):
    tools = [t for t in spec.SCENE_TOOLS if CFG.images or not t.get("image")]
    by_name = {t["name"]: t for t in tools}
    functions = [_function(t["name"], t["description"], t["schema"]) for t in tools]
    functions.append(_function("submit_verdict", "Hand in the final verdict. Call it once, when you are done.",
                               schema))
    if not CFG.images:
        system += ("\n\nNo figures are available in this run: judge from the overview, the time series and the "
                   "nearby road users.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    calls, texts, usage, errors = [], [], {"prompt_tokens": 0, "completion_tokens": 0}, []
    names = [t["name"] for t in tools] + ["submit_verdict"]
    for turn in range(1, max_turns + 1):
        last = turn == max_turns
        resp = await _chat(model, messages, functions,
                           {"type": "function", "function": {"name": "submit_verdict"}} if last else None)
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0) or 0
        msg = resp["choices"][0]["message"]
        content = _strip(msg.get("content"))
        if content:
            texts.append(content)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            verdict = _json_in(content)
            if verdict is not None:
                verdict = _coerce(verdict, schema)
                problems = spec.validate(verdict, schema)
                if not problems:
                    return _record(model, names, verdict, "success", turn, calls, texts, usage, errors)
                errors.append("; ".join(problems))
                messages.append({"role": "user", "content": "That verdict is invalid: %s. Call submit_verdict with "
                                                            "a corrected one." % "; ".join(problems)})
            else:
                messages.append({"role": "user", "content": "Continue with the tools, or call submit_verdict if you "
                                                            "are done."})
            continue
        for n, tc in enumerate(tool_calls):
            tc.setdefault("id", "call_%d_%d" % (turn, n))
            tc.setdefault("type", "function")
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        figures, verdict = [], None
        for tc in tool_calls:
            fname = tc["function"]["name"]
            try:
                args = _arguments(tc["function"].get("arguments"))
            except json.JSONDecodeError as exc:
                args, reply = None, "the arguments are not valid JSON: %s" % exc
            calls.append({"tool": fname, "input": args})
            if args is None:
                pass
            elif fname == "submit_verdict":
                args = _coerce(args, schema)
                problems = spec.validate(args, schema)
                if problems:
                    errors.append("; ".join(problems))
                    reply = "rejected: %s. Call submit_verdict again with these fixed." % "; ".join(problems)
                else:
                    verdict, reply = args, "accepted"
            elif fname in by_name:
                try:
                    kind, out = by_name[fname]["call"](scene, args, fig_dir)
                    if kind == "image":
                        figures.append(out)
                        reply = "figure %d follows in the next message" % len(figures)
                    else:
                        reply = out
                except Exception as exc:  # the agent sees the error and can correct its call
                    reply = "%s: %s" % (type(exc).__name__, exc)
            else:
                reply = "no tool named %s; the tools are %s" % (fname, ", ".join(names))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": reply})
        if verdict is not None:
            return _record(model, names, verdict, "success", turn, calls, texts, usage, errors)
        if figures:
            parts = [{"type": "text", "text": "Figure(s) from render_scene:"}]
            parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f).decode()}}
                      for f in figures]
            messages.append({"role": "user", "content": parts})
            _trim_images(messages, CFG.max_images)
    return _record(model, names, None, "error_max_turns", max_turns, calls, texts, usage, errors)


async def review(name, rules, model, fig_dir, budget=None, max_turns=24):
    return await _scene_agent(name, spec.REVIEWER.format(rules=rules), spec.review_prompt(name), spec.VERDICT,
                              model, fig_dir, max_turns)


async def criticise(name, first_verdict, rules, model, fig_dir, budget=None, max_turns=24):
    return await _scene_agent(name, spec.CRITIC.format(rules=rules), spec.critic_prompt(name, first_verdict),
                              spec.CRITIC_VERDICT, model, fig_dir, max_turns)


# --- the rule auditor ------------------------------------------------------------------------

def _file_tools(roots):
    """Read-only file tools limited to ``roots`` (the skill first, then the run folder)."""

    def where(path):
        if not spec.inside(roots, path or "."):
            raise PermissionError("only the skill and the run folder may be read")
        p = Path(path or ".")
        return (roots[0] / p if not p.is_absolute() else p).resolve()

    def list_files(a):
        base = where(a.get("path"))
        found = [str(p.relative_to(base)) for p in sorted(base.rglob("*")) if p.is_file()
                 and fnmatch.fnmatch(p.name, a.get("pattern", "*")) and "__pycache__" not in p.parts]
        return "\n".join(found[:300]) or "no files"

    def read_file(a):
        lines = where(a["path"]).read_text(encoding="utf-8", errors="replace").splitlines()
        start = int(a.get("offset", 1))
        chunk = lines[start - 1:start - 1 + int(a.get("limit", 400))]
        return "\n".join("%d\t%s" % (start + k, line) for k, line in enumerate(chunk)) or "(empty)"

    def grep(a):
        base, pattern = where(a.get("path")), re.compile(a["pattern"])
        files = [base] if base.is_file() else [p for p in sorted(base.rglob("*")) if p.is_file()
                                               and p.suffix in (".py", ".md", ".csv", ".json")]
        out = []
        for f in files:
            for k, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    out.append("%s:%d: %s" % (f, k, line[:300]))
        return "\n".join(out[:200]) or "no matches"

    str_ = {"type": "string"}
    return {
        "list_files": (list_files, "List files under a folder (default: the skill), optionally matching a glob "
                                   "pattern on the file name.",
                       {"type": "object", "properties": {"path": str_, "pattern": str_}}),
        "read_file": (read_file, "Read a text file with line numbers (offset is 1-based, limit lines).",
                      {"type": "object", "properties": {"path": str_, "offset": {"type": "number"},
                                                        "limit": {"type": "number"}}, "required": ["path"]}),
        "grep": (grep, "Search .py/.md/.csv/.json files under a folder or in one file for a regular expression.",
                 {"type": "object", "properties": {"pattern": str_, "path": str_}, "required": ["pattern"]}),
    }


async def audit(run_dir, skill_dir, model, budget=None, max_turns=40):
    run_dir, skill_dir = Path(run_dir).resolve(), Path(skill_dir).resolve()
    tools = _file_tools([skill_dir, run_dir])
    functions = [_function(n, d, s) for n, (_, d, s) in tools.items()]
    messages = [{"role": "system", "content": spec.AUDITOR}, {"role": "user", "content": spec.audit_prompt(run_dir)}]
    calls, texts, usage = [], [], {"prompt_tokens": 0, "completion_tokens": 0}
    for turn in range(1, max_turns + 1):
        resp = await _chat(model, messages, functions, "none" if turn == max_turns else None)
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k, 0) or 0
        msg = resp["choices"][0]["message"]
        content = _strip(msg.get("content"))
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            out = _record(model, list(tools), None, "success", turn, calls, texts + [content], usage, final_text=content)
            out["markdown"] = content
            return out
        if content:
            texts.append(content)
        for n, tc in enumerate(tool_calls):
            tc.setdefault("id", "call_%d_%d" % (turn, n))
            tc.setdefault("type", "function")
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc in tool_calls:
            fname = tc["function"]["name"]
            try:
                args = _arguments(tc["function"].get("arguments"))
                reply = tools[fname][0](args) if fname in tools else "no tool named %s" % fname
            except Exception as exc:
                args, reply = None, "%s: %s" % (type(exc).__name__, exc)
            calls.append({"tool": fname, "input": args})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": reply[:20000]})
    out = _record(model, list(tools), None, "error_max_turns", max_turns, calls, texts, usage)
    out["markdown"] = "\n\n".join(texts)
    return out
