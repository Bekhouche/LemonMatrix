"""Built-in prompt sets and prompt-file loading for router runs.

A router only shows its behaviour on a mix of requests: one fixed prompt always
matches the same rule. ``router-mix`` pairs short everyday questions with
coding and long-form requests, the split a local-first policy is meant to make.
"""

from __future__ import annotations

import json
from pathlib import Path

ROUTER_MIX = [
    "What is the capital of Canada?",
    "Give me three name ideas for a lemonade stand.",
    "Translate 'good morning, how are you?' into French.",
    "What is 17 multiplied by 23?",
    "Suggest a short title for a blog post about houseplants.",
    "Write a Python function that parses an ISO 8601 date string and returns a datetime, "
    "with error handling and unit tests.",
    "Explain this stack trace and how to fix it: "
    "TypeError: 'NoneType' object is not subscriptable at line 42 in handler.py.",
    "Refactor this JavaScript function to use async/await: "
    "function load(url, cb) { fetch(url).then(r => r.json()).then(d => cb(null, d)).catch(cb); }",
    "Compare the CAP theorem trade-offs of a leader-based replicated database and a leaderless "
    "one, with a concrete failure scenario for each, in a detailed answer.",
    "Write a SQL query that returns the top five customers by total order value in the last "
    "90 days, and explain the indexes it needs.",
]

PROMPT_SETS = {"router-mix": ROUTER_MIX}


def load_prompt_file(path: str | Path) -> list[str]:
    """Read a prompt set: JSONL (strings or {"prompt": ...}) or one prompt per line."""
    path = Path(path)
    text = path.read_text()
    prompts: list[str] = []
    if path.suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") if isinstance(item, dict) else item
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_no}: expected a string or an object with a 'prompt' string")
            prompts.append(prompt)
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts
