#!/usr/bin/env python3
"""IL state/option encoder + dataset builder.

Encodes each in-game decision (obs, picked_option_indices) into arrays ready
for a small MLP with card/attack embeddings.

Per decision (acting player = current.yourIndex):
  state_scalars   : float32[...]   turn/counts/hp/context/minMaxCount/flags
  slot_card_ids   : int32[12]      my active(1)+bench(5)+opp active(1)+opp bench(5)
  slot_scalars    : float32[12*4]  hp_ratio, n_energies, n_tools, appearThisTurn
  hand_card_ids   : int32[20]      my hand card ids (pad 0)
  stadium_id      : int32          stadium card id (0 if none)
  option_type     : int32[N]       OptionType
  option_area     : int32[N]       AreaType of referenced card
  option_card_id  : int32[N]       referenced card id (0 if not resolvable)
  option_target_id: int32[N]       target pokemon card id (ATTACH/EVOLVE)
  option_attack_id: int32[N]       attack id (ATTACK)
  option_scalars  : float32[N*2]   count, inPlayIndex
  picked          : int8[N]        1 if option was chosen
  max_count       : int            select.maxCount

Build dataset: python tools/il_encode.py <zip> out.npz
"""
import sys, os, json, zipfile, argparse
import numpy as np

DRAGAPULT_EX = 121
N_HAND_PAD = 20
N_BENCH = 5
N_SLOTS = 2 * (1 + N_BENCH)  # my active+bench, opp active+bench
N_OPTION_SCALARS = 2

# AreaType (from cg/api.py)
HAND, DISCARD, ACTIVE, BENCH, PRIZE, DECK = 2, 3, 4, 5, 6, 1


def _card_id(card):
    return card.get("id", 0) if isinstance(card, dict) else 0


def _zone_cards(player, area, index):
    """Return the card/pokemon dict at (area, index) for a player, or None."""
    if area == HAND:
        z = player.get("hand", [])
    elif area == DISCARD:
        z = player.get("discard", [])
    elif area == ACTIVE:
        z = player.get("active", [])
    elif area == BENCH:
        z = player.get("bench", [])
    elif area == PRIZE:
        z = player.get("prize", [])
    else:
        return None
    if isinstance(z, list) and 0 <= index < len(z):
        return z[index]
    return None


def resolve_option(obs, pi, opt):
    """Return (card_id, target_id, attack_id) for an option."""
    t = opt.get("type", 0)
    cur = obs["current"]
    players = cur["players"]
    opt_pi = opt.get("playerIndex", pi)
    player = players[opt_pi] if 0 <= opt_pi < len(players) else {}
    card_id = target_id = attack_id = 0
    area = opt.get("area", 0)
    index = opt.get("index", 0)
    # the primary card being selected/played
    if t == 7:  # PLAY: index in acting player's hand
        player = players[pi] if 0 <= pi < len(players) else {}
        h = player.get("hand", [])
        if 0 <= index < len(h):
            card_id = _card_id(h[index])
    elif t in (3, 4, 5, 10, 11):  # CARD/TOOL_CARD/ENERGY_CARD/ABILITY/DISCARD: area+index+playerIndex
        c = _zone_cards(player, area, index)
        card_id = _card_id(c) if c else 0
    elif t == 8 or t == 9:  # ATTACH/EVOLVE: card at (area,index) ; target at (inPlayArea,inPlayIndex)
        c = _zone_cards(player, area, index)
        card_id = _card_id(c) if c else 0
        tgt_pi = opt.get("playerIndex", pi)
        tgt_player = players[tgt_pi] if 0 <= tgt_pi < len(players) else {}
        tgt = _zone_cards(tgt_player, opt.get("inPlayArea", 0), opt.get("inPlayIndex", 0))
        target_id = _card_id(tgt) if tgt else 0
    elif t == 6:  # ENERGY: pokemon at (area,index,playerIndex)
        c = _zone_cards(player, area, index)
        card_id = _card_id(c) if c else 0
    elif t == 13:  # ATTACK
        attack_id = opt.get("attackId", 0)
    elif t == 15:  # SKILL: cardId field
        card_id = opt.get("cardId", 0)
    return card_id, target_id, attack_id


def _slot_scalars(pkm):
    if not isinstance(pkm, dict):
        return [0.0, 0.0, 0.0, 0.0]
    hp = pkm.get("hp", 0); max_hp = pkm.get("maxHp", 0) or 1
    return [
        hp / max_hp,
        float(len(pkm.get("energies", []))),
        float(len(pkm.get("tools", []))),
        1.0 if pkm.get("appearThisTurn") else 0.0,
    ]


def encode_decision(obs, action):
    """Encode one in-game decision. Returns dict of arrays or None if skip."""
    sel = obs.get("select")
    if not sel:
        return None
    options = sel.get("option", [])
    if not options:
        return None
    max_count = sel.get("maxCount", 1) or 1
    cur = obs["current"]
    pi = cur.get("yourIndex", 0)
    players = cur.get("players", [])
    if pi is None or pi >= len(players):
        return None
    me = players[pi]
    opp = players[1 - pi] if len(players) > 1 else {}

    # ---- state scalars ----
    my_active = (me.get("active") or [None])
    opp_active = (opp.get("active") or [None])
    my_a = my_active[0] or {}
    op_a = opp_active[0] or {}
    ctx = sel.get("context", 0)
    ctx_oh = np.zeros(49, dtype=np.float32)  # SelectContext up to ~48
    if 0 <= ctx < 49:
        ctx_oh[ctx] = 1.0
    state_scalars = np.array([
        cur.get("turn", 0),
        me.get("handCount", 0), me.get("deckCount", 0), len(me.get("prize", [])),
        opp.get("handCount", 0), opp.get("deckCount", 0), len(opp.get("prize", [])),
        my_a.get("hp", 0), my_a.get("maxHp", 0) or 1, len(my_a.get("energies", [])), len(my_a.get("tools", [])),
        op_a.get("hp", 0), op_a.get("maxHp", 0) or 1, len(op_a.get("energies", [])), len(op_a.get("tools", [])),
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("stadiumPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("retreated") else 0.0,
        1.0 if cur.get("firstPlayer") == pi else 0.0,
        sel.get("minCount", 0), max_count,
        sel.get("type", 0),
    ], dtype=np.float32)
    state_scalars = np.concatenate([state_scalars, ctx_oh])

    # ---- card-id slots (my active, my bench, opp active, opp bench) ----
    slot_ids = []
    slot_sc = []
    for pkm in [my_a] + (me.get("bench") or [])[:N_BENCH] + [op_a] + (opp.get("bench") or [])[:N_BENCH]:
        slot_ids.append(pkm.get("id", 0) if isinstance(pkm, dict) else 0)
        slot_sc.extend(_slot_scalars(pkm))
    while len(slot_ids) < N_SLOTS:
        slot_ids.append(0); slot_sc.extend([0, 0, 0, 0])
    slot_card_ids = np.array(slot_ids[:N_SLOTS], dtype=np.int32)
    slot_scalars = np.array(slot_sc[:N_SLOTS * 4], dtype=np.float32)

    # ---- hand card ids (pad) ----
    hand_ids = [_card_id(c) for c in (me.get("hand") or [])][:N_HAND_PAD]
    while len(hand_ids) < N_HAND_PAD:
        hand_ids.append(0)
    hand_card_ids = np.array(hand_ids, dtype=np.int32)

    stadium_id = np.int32(_card_id(cur.get("stadium")))

    # ---- options ----
    n = len(options)
    o_type = np.zeros(n, dtype=np.int32)
    o_area = np.zeros(n, dtype=np.int32)
    o_card = np.zeros(n, dtype=np.int32)
    o_tgt = np.zeros(n, dtype=np.int32)
    o_atk = np.zeros(n, dtype=np.int32)
    o_sc = np.zeros((n, N_OPTION_SCALARS), dtype=np.float32)
    picked = np.zeros(n, dtype=np.int8)
    picked_set = set(action) if isinstance(action, (list, tuple)) else set()
    for i, opt in enumerate(options):
        o_type[i] = opt.get("type", 0)
        o_area[i] = opt.get("area", 0)
        cid, tid, aid = resolve_option(obs, pi, opt)
        o_card[i] = cid
        o_tgt[i] = tid
        o_atk[i] = aid
        o_sc[i, 0] = opt.get("count", 0)
        o_sc[i, 1] = opt.get("inPlayIndex", 0)
        if i in picked_set:
            picked[i] = 1
    return {
        "state_scalars": state_scalars,
        "slot_card_ids": slot_card_ids,
        "slot_scalars": slot_scalars,
        "hand_card_ids": hand_card_ids,
        "stadium_id": stadium_id,
        "option_type": o_type, "option_area": o_area,
        "option_card_id": o_card, "option_target_id": o_tgt,
        "option_attack_id": o_atk, "option_scalars": o_sc,
        "picked": picked, "max_count": max_count, "ctx": ctx,
    }


def player_decks(game):
    steps = game["steps"]
    if len(steps) < 2:
        return None, None
    out = []
    for pi in range(2):
        act = steps[1][pi].get("action") if pi < len(steps[1]) else None
        out.append(act if isinstance(act, list) and len(act) == 60 else None)
    return out[0], out[1]


def iter_games(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            try:
                yield json.loads(z.read(name))
            except Exception:
                continue


def extract_dragapult_pairs(game):
    """Yield (obs, action) for Dragapult players' in-game decisions."""
    d0, d1 = player_decks(game)
    steps = game["steps"]
    for pi, deck in enumerate((d0, d1)):
        if deck is None or DRAGAPULT_EX not in deck:
            continue
        for t in range(1, len(steps) - 1):
            obs = steps[t][pi].get("observation")
            if not isinstance(obs, dict) or obs.get("select") is None:
                continue
            nxt = steps[t + 1][pi].get("action")
            if not isinstance(nxt, list) or not nxt:
                continue
            n_opt = len(obs["select"].get("option", []))
            if n_opt and max(nxt) >= n_opt:
                continue
            yield obs, nxt


def build(zip_path, out_path, max_games=0):
    records = []
    n_games = n_draga = n_pairs = 0
    for game in iter_games(zip_path):
        if max_games and n_games >= max_games:
            break
        n_games += 1
        for obs, act in extract_dragapult_pairs(game):
            rec = encode_decision(obs, act)
            if rec is not None:
                records.append(rec)
                n_pairs += 1
        d0, d1 = player_decks(game)
        if (d0 and DRAGAPULT_EX in d0) or (d1 and DRAGAPULT_EX in d1):
            n_draga += 1
    print(f"games={n_games} dragapult_games={n_draga} pairs={n_pairs}")
    if not records:
        return
    # serialize as a pickle of the list (variable-size option arrays)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"saved {len(records)} decisions -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("out", help="output .pkl path")
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()
    build(args.zip, args.out, args.max_games)
