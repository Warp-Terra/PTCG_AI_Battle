#!/usr/bin/env python3
"""Local cabt match runner.

Runs two agents against each other in the official `cabt` environment over N
games, alternating seats, and reports win/loss/draw + per-agent errors.

How it works:
  - The `cg` engine is provided via `../cg-lib` on sys.path (the agent's
    `from cg.api import ...` resolves there at module-load time).
  - Agents are loaded as callables and wrapped so that:
      * the wrapper's signature is `(observation, configuration=None)` —
        kaggle_environments' `Agent.act` truncates call args to the callable's
        `co_argcount`, so a bare `*args` wrapper (co_argcount=0) would receive
        zero args. An explicit 2-param signature receives both, then we forward
        based on the real agent's argcount;
      * the working directory is chdir'd to the agent's own directory during
        the call, so a cwd-relative `deck.csv` read (as the official sample and
        our baseline do) resolves per-agent — this is what makes agents in
        *different* directories play each other (cabt runs both in-process with
        a single shared cwd otherwise).
  - Deck reads only happen at the initial selection (once per agent), and cabt
    calls agents sequentially (turn-based), so the chdir/restore is race-free.

Usage:
    python tools/run_match.py <agent_a_dir> <agent_b_dir> [games]
    python tools/run_match.py data/sample_submission data/sample_submission 4
    python tools/run_match.py agents/baseline agents/baseline 10
"""
import os
import sys
import warnings
import logging
import argparse

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cg-lib"))


def load_agent(agent_dir: str):
    """Load the `agent` callable from <agent_dir>/main.py and wrap it.

    The wrapper chdir's into the agent dir during each call (so a cwd-relative
    deck.csv read resolves) and forwards the observation respecting the real
    agent's positional argcount (1 or 2).
    """
    agent_dir = os.path.abspath(agent_dir)
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"no main.py in {agent_dir}")
    raw = open(main_py, "r").read()
    from kaggle_environments.agent import get_last_callable
    cb = get_last_callable(raw, path=main_py)
    if cb is None:
        raise RuntimeError(f"no callable `agent` found in {main_py}")
    cb_argcount = getattr(getattr(cb, "__code__", None), "co_argcount", 1) or 1

    def wrapped(observation, configuration=None):
        prev = os.getcwd()
        os.chdir(agent_dir)
        try:
            if cb_argcount >= 2:
                return cb(observation, configuration)
            return cb(observation)
        finally:
            os.chdir(prev)

    return wrapped


def run_game(a, b):
    """Run one cabt game with a=seat0, b=seat1. Returns (reward_a, reward_b, status_a, status_b)."""
    from kaggle_environments import make
    env = make("cabt")
    res = env.run([a, b])
    last = res[-1]
    return (last[0].get("reward"), last[1].get("reward"),
            last[0].get("status"), last[1].get("status"))


def main():
    ap = argparse.ArgumentParser(description="Run two cabt agents over N games.")
    ap.add_argument("agent_a", help="agent A directory (contains main.py + deck.csv)")
    ap.add_argument("agent_b", help="agent B directory (contains main.py + deck.csv)")
    ap.add_argument("games", type=int, nargs="?", default=10, help="number of games (default 10)")
    args = ap.parse_args()

    a = load_agent(args.agent_a)
    b = load_agent(args.agent_b)

    # tally: a_win, b_win, draw, a_err, b_err
    tally = [0, 0, 0, 0, 0]
    for g in range(args.games):
        seats = (a, b) if g % 2 == 0 else (b, a)
        a_is_seat0 = g % 2 == 0
        try:
            ra, rb, sa, sb = run_game(*seats)
        except Exception as e:
            print(f"  game {g+1}/{args.games}: engine crash: {e!r}")
            tally[3 if a_is_seat0 else 4] += 1
            continue
        a_err = sa in ("ERROR", "INVALID") or ra is None
        b_err = sb in ("ERROR", "INVALID") or rb is None
        if a_err and b_err:
            tally[2] += 1
        elif a_err:
            tally[3 if a_is_seat0 else 4] += 1
        elif b_err:
            tally[4 if a_is_seat0 else 3] += 1
        elif ra > rb:
            tally[0] += 1
        elif rb > ra:
            tally[1] += 1
        else:
            tally[2] += 1
        flag = " [error]" if (a_err or b_err) else ""
        print(f"  game {g+1}/{args.games}: A={ra} B={rb} status=({sa},{sb}){flag}")

    aw, bw, dw, ae, be = tally
    print(f"\n[result] A={args.agent_a}  vs  B={args.agent_b}  over {args.games} games")
    print(f"  A wins: {aw}  |  B wins: {bw}  |  draws: {dw}  |  A errors: {ae}  |  B errors: {be}")
    decisive = aw + bw
    if decisive:
        print(f"  A win-rate (decisive): {aw/decisive*100:.1f}%  ({aw}/{decisive})")


if __name__ == "__main__":
    main()
