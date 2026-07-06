#!/usr/bin/env python3
"""Parse episode zip(s) into (obs, action) pairs for IL.

Filters games where a player's deck contains Dragapult ex (card 121), and
extracts that player's in-game decisions as (obs_dict, picked_option_indices).

Episode JSON structure (verified on 2026-06-16):
  - rewards: [p0, p1], winner = higher.
  - info.Agents[i].Name / info.TeamNames: player identities.
  - steps[t][pi]: {action, observation, reward, status, ...}.
  - step[0]: no action. step[1][pi].action = the 60-card deck.
  - off-by-one: obs@step[t] -> action@step[t+1] (the answer).
  - obs['select'] is None during deck-selection; non-None in-game.

Usage:
    python tools/parse_episodes.py <zip> [--serialize out.npz] [--max-games N]
"""
import sys, os, json, zipfile, argparse, collections

DRAGAPULT_EX = 121  # Dragapult ex card id; presence => Dragapult deck


def iter_games(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            try:
                yield json.loads(z.read(name))
            except Exception:
                continue


def player_decks(game):
    """Return (deck0, deck1) = each player's 60-card deck from step[1]."""
    steps = game["steps"]
    if len(steps) < 2:
        return None, None
    out = []
    for pi in range(2):
        act = steps[1][pi].get("action") if pi < len(steps[1]) else None
        out.append(act if isinstance(act, list) and len(act) == 60 else None)
    return out[0], out[1]


def extract_pairs(game, pi):
    """Yield (obs_dict, action_indices, context) for player pi's in-game decisions."""
    steps = game["steps"]
    for t in range(1, len(steps) - 1):
        obs = steps[t][pi].get("observation")
        if not isinstance(obs, dict):
            continue
        sel = obs.get("select")
        if sel is None:
            continue  # deck-selection or no decision
        nxt = steps[t + 1][pi].get("action")
        if not isinstance(nxt, list) or len(nxt) == 0:
            continue  # inactive / no-op
        # sanity: action indices must be within option range
        n_opt = len(sel.get("option", []))
        if n_opt and max(nxt) >= n_opt:
            continue
        ctx = sel.get("context")
        yield obs, nxt, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--serialize", help="optional output .npz path (raw pickled pairs)")
    args = ap.parse_args()

    n_games = 0
    n_draga_games = 0
    n_pairs = 0
    n_wins = 0
    ctx_counts = collections.Counter()
    sample = None
    pairs_for_save = []

    for gi, game in enumerate(iter_games(args.zip)):
        if args.max_games and gi >= args.max_games:
            break
        n_games += 1
        d0, d1 = player_decks(game)
        rewards = game.get("rewards", [0, 0])
        for pi, deck in enumerate((d0, d1)):
            if deck is None or DRAGAPULT_EX not in deck:
                continue
            if pi == 0:
                n_draga_games += 1  # count once per game with a Dragapult player
            for obs, act, ctx in extract_pairs(game, pi):
                n_pairs += 1
                ctx_counts[ctx] += 1
                if sample is None:
                    sample = (obs, act, ctx)
                if args.serialize:
                    pairs_for_save.append((obs, act, ctx))
            # win for this dragapult player?
            other = 1 - pi
            if rewards[pi] is not None and rewards[other] is not None and rewards[pi] > rewards[other]:
                n_wins += 1

    print(f"games total: {n_games}")
    print(f"games with a Dragapult player: {n_draga_games}")
    print(f"Dragapult in-game decision pairs: {n_pairs}")
    print(f"Dragapult player wins (across appearances): {n_wins}")
    print(f"context distribution (top 12): {ctx_counts.most_common(12)}")
    if sample:
        obs, act, ctx = sample
        print(f"sample pair: ctx={ctx} action={act} n_options={len(obs['select'].get('option',[]))} maxCount={obs['select'].get('maxCount')}")

    if args.serialize and pairs_for_save:
        import pickle
        with open(args.serialize, "wb") as f:
            pickle.dump(pairs_for_save, f)
        print(f"serialized {len(pairs_for_save)} pairs -> {args.serialize}")


if __name__ == "__main__":
    main()
