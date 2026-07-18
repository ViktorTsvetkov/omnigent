"""Deterministic psmux CLI double for backend conformance tests."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

STATE_DIR_ENV_VAR = "OMNIGENT_FAKE_PSMUX_STATE_DIR"
EXIT_SENTINEL = "__omnigent_fake_psmux_exit__"
SUBMIT_SENTINEL = "[fake-psmux:submit]"
FINAL_SCREEN_MARKER = "[fake-psmux:inner-exited]\n"


def key_marker(key: str) -> str:
    return f"<{key}>"


def _state_path(namespace: str) -> Path:
    root = Path(os.environ[STATE_DIR_ENV_VAR])
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(namespace.encode()).hexdigest()
    return root / f"{digest}.json"


def _load(namespace: str) -> dict[str, object] | None:
    try:
        return json.loads(_state_path(namespace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save(namespace: str, state: dict[str, object]) -> None:
    _state_path(namespace).write_text(json.dumps(state), encoding="utf-8")


def _remove(namespace: str) -> None:
    with contextlib.suppress(OSError):
        _state_path(namespace).unlink()


def _positionals(args: list[str]) -> list[str]:
    values: list[str] = []
    index = 1
    while index < len(args):
        token = args[index]
        if token == "-l":
            index += 1
        elif token in {"-t", "-N"}:
            index += 2
        else:
            values.append(token)
            index += 1
    return values


def _repeat_count(args: list[str]) -> int:
    if "-N" not in args:
        return 1
    try:
        return int(args[args.index("-N") + 1])
    except (IndexError, ValueError):
        return 1


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print("tmux 3.3.6")
        return 0

    namespace = "default"
    config_path: Path | None = None
    index = 0
    while index < len(argv):
        if argv[index] == "-L":
            namespace = argv[index + 1]
            index += 2
        elif argv[index] == "-f":
            config_path = Path(argv[index + 1])
            index += 2
        else:
            break
    args = argv[index:]
    if not args:
        return 0
    command = args[0]

    if command == "new-session":
        inner = args[args.index("--") + 1 :] if "--" in args else []
        exits = inner == [EXIT_SENTINEL]
        keep_alive = False
        if config_path is not None:
            with contextlib.suppress(OSError):
                keep_alive = "remain-on-exit on" in config_path.read_text(encoding="utf-8")
        if exits and not keep_alive:
            _remove(namespace)
        else:
            _save(
                namespace,
                {
                    "screen": FINAL_SCREEN_MARKER if exits else "",
                    "pane_dead": exits,
                },
            )
        return 0

    state = _load(namespace)
    if command in {"list-panes", "capture-pane", "send-keys", "display-message"}:
        if state is None:
            return 1

    if command == "list-panes":
        print("1" if state and state.get("pane_dead") else "0")
    elif command == "capture-pane":
        sys.stdout.write(str(state.get("screen", "")))
    elif command == "send-keys":
        assert state is not None
        screen = str(state.get("screen", ""))
        values = _positionals(args)
        if "-l" in args:
            screen += "".join(values)
        else:
            for _ in range(_repeat_count(args)):
                for key in values:
                    screen += key_marker(key)
                    if key == "Enter":
                        screen += SUBMIT_SENTINEL
        state["screen"] = screen
        _save(namespace, state)
    elif command == "display-message":
        print("0,0")
    elif command == "kill-server":
        existed = state is not None
        _remove(namespace)
        return 0 if existed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
