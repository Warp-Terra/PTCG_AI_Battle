import os
import random
import sys
import traceback

from cg.api import Observation, to_observation_class


def _deck_path() -> str:
    """Resolve deck.csv without relying on __file__ (cabt execs agents without it).

    Search order: cwd, each sys.path entry, then the Kaggle submission layout.
    kaggle_environments adds the agent's own directory to sys.path, so this finds
    the deck co-located with main.py regardless of the process cwd.
    """
    for p in ["."] + list(sys.path) + ["/kaggle_simulations/agent"]:
        if not p:
            continue
        cand = os.path.join(p, "deck.csv")
        if os.path.exists(cand):
            return cand
    return "deck.csv"


def read_deck_csv() -> list[int]:
    """Read 60 card IDs from deck.csv."""
    with open(_deck_path(), "r") as f:
        lines = f.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(lines[i]))
    return deck


def _legal_fallback(obs: Observation) -> list[int]:
    """Always-legal selection: the first maxCount option indices."""
    n = len(obs.select.option)
    k = obs.select.maxCount
    return list(range(min(k, n)))


def agent(obs_dict: dict) -> list[int]:
    """Pokémon TCG baseline agent (random legal selection, never crashes).

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
    except Exception:
        traceback.print_exc()
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return read_deck_csv()
            return _legal_fallback(obs)
        except Exception:
            traceback.print_exc()
            return [0]
