"""Interactive child process used by the native ConPTY conformance tests."""

from __future__ import annotations

import argparse
import msvcrt
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--exit", action="store_true")
args = parser.parse_args()

if args.exit:
    print("CONPTY_FINAL_SCREEN", flush=True)
    raise SystemExit(0)

sys.stdout.write("\x1b[2J\x1b[HCONPTY_READY\r\n")
sys.stdout.flush()

paste = False
escape = ""
while True:
    char = msvcrt.getwch()
    if escape:
        escape += char
        if escape == "\x1b[200~":
            paste = True
            escape = ""
        elif escape == "\x1b[201~":
            paste = False
            escape = ""
        elif "\x1b[200~".startswith(escape) or "\x1b[201~".startswith(escape):
            continue
        else:
            sys.stdout.write("<KEY:Escape>")
            for pending in escape[1:]:
                sys.stdout.write("<KEY:Tab>" if pending == "\t" else pending)
            escape = ""
    elif char == "\x1b":
        escape = char
    elif char == "\r" and not paste:
        sys.stdout.write("\r\nCONPTY_SUBMITTED\r\n")
    elif char == "\t" and not paste:
        sys.stdout.write("<KEY:Tab>")
    elif char in {"\x08", "\x7f"} and not paste:
        sys.stdout.write("<KEY:BSpace>")
    elif char == "\n":
        sys.stdout.write("\r\n")
    else:
        sys.stdout.write(char)
    sys.stdout.flush()
