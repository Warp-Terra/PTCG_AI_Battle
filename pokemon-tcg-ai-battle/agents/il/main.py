import os
import sys
import json
import numpy as np

# === asset (deck.csv / model.npz) resolution: cwd -> sys.path -> Kaggle layout ===
def _asset_path(name):
    for p in ["."] + list(sys.path) + ["/kaggle_simulations/agent"]:
        if p and os.path.exists(os.path.join(p, name)):
            return os.path.join(p, name)
    return name

with open(_asset_path("deck.csv"), "r") as f:
    _lines = f.read().split("\n")
my_deck = [int(_lines[i]) for i in range(60)]

_W = np.load(_asset_path("model.npz"))
W_card = _W["card_emb.weight"]      # (1268, 64)
W_atk = _W["atk_emb.weight"]        # (1557, 64)
# state_mlp: 0=Linear(1016,256), 2=Linear(256,256)
S0W, S0B = _W["state_mlp.0.weight"], _W["state_mlp.0.bias"]
S2W, S2B = _W["state_mlp.2.weight"], _W["state_mlp.2.bias"]
# opt_mlp: 0=Linear(223,256), 2=Linear(256,128)
O0W, O0B = _W["opt_mlp.0.weight"], _W["opt_mlp.0.bias"]
O2W, O2B = _W["opt_mlp.2.weight"], _W["opt_mlp.2.bias"]
# score: 0=Linear(384,128), 2=Linear(128,1)
C0W, C0B = _W["score.0.weight"], _W["score.0.bias"]
C2W, C2B = _W["score.2.weight"], _W["score.2.bias"]

N_BENCH, N_SLOTS, N_HAND_PAD = 5, 12, 20
HAND, DISCARD, ACTIVE, BENCH, PRIZE, DECK = 2, 3, 4, 5, 6, 1

def _cid(c): return c.get("id", 0) if isinstance(c, dict) else 0

def _zone(player, area, index):
    if area == HAND: z = player.get("hand", [])
    elif area == DISCARD: z = player.get("discard", [])
    elif area == ACTIVE: z = player.get("active", [])
    elif area == BENCH: z = player.get("bench", [])
    elif area == PRIZE: z = player.get("prize", [])
    else: return None
    return z[index] if isinstance(z, list) and 0 <= index < len(z) else None

def _resolve(obs, pi, opt):
    t = opt.get("type", 0)
    players = obs["current"]["players"]
    opt_pi = opt.get("playerIndex", pi)
    player = players[opt_pi] if 0 <= opt_pi < len(players) else {}
    cid = tid = aid = 0
    area = opt.get("area", 0); index = opt.get("index", 0)
    if t == 7:  # PLAY
        player = players[pi] if 0 <= pi < len(players) else {}
        h = player.get("hand", [])
        cid = _cid(h[index]) if 0 <= index < len(h) else 0
    elif t in (3, 4, 5, 10, 11):
        c = _zone(player, area, index); cid = _cid(c) if c else 0
    elif t in (8, 9):
        c = _zone(player, area, index); cid = _cid(c) if c else 0
        tgt = _zone(player, opt.get("inPlayArea", 0), opt.get("inPlayIndex", 0))
        tid = _cid(tgt) if tgt else 0
    elif t == 6:
        c = _zone(player, area, index); cid = _cid(c) if c else 0
    elif t == 13:
        aid = opt.get("attackId", 0)
    elif t == 15:
        cid = opt.get("cardId", 0)
    return cid, tid, aid

def _slot_sc(pkm):
    if not isinstance(pkm, dict): return [0.0, 0.0, 0.0, 0.0]
    return [pkm.get("hp", 0) / (pkm.get("maxHp", 0) or 1),
            float(len(pkm.get("energies", []))),
            float(len(pkm.get("tools", []))),
            1.0 if pkm.get("appearThisTurn") else 0.0]

def _relu(x): return np.maximum(x, 0.0)

def _encode(obs):
    sel = obs.get("select")
    if not sel: return None
    options = sel.get("option", [])
    if not options: return None
    cur = obs["current"]; pi = cur.get("yourIndex", 0)
    players = cur.get("players", [])
    if pi is None or pi >= len(players): return None
    me = players[pi]; opp = players[1 - pi] if len(players) > 1 else {}
    my_a = (me.get("active") or [{}])[0] or {}
    op_a = (opp.get("active") or [{}])[0] or {}
    ctx = sel.get("context", 0)
    ctx_oh = np.zeros(49, dtype=np.float32)
    if 0 <= ctx < 49: ctx_oh[ctx] = 1.0
    base = np.array([
        cur.get("turn", 0), me.get("handCount", 0), me.get("deckCount", 0), len(me.get("prize", [])),
        opp.get("handCount", 0), opp.get("deckCount", 0), len(opp.get("prize", [])),
        my_a.get("hp", 0), my_a.get("maxHp", 0) or 1, len(my_a.get("energies", [])), len(my_a.get("tools", [])),
        op_a.get("hp", 0), op_a.get("maxHp", 0) or 1, len(op_a.get("energies", [])), len(op_a.get("tools", [])),
        1.0 if cur.get("supporterPlayed") else 0.0, 1.0 if cur.get("stadiumPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0, 1.0 if cur.get("retreated") else 0.0,
        1.0 if cur.get("firstPlayer") == pi else 0.0, sel.get("minCount", 0), sel.get("maxCount", 1) or 1,
        sel.get("type", 0),
    ], dtype=np.float32)
    state_scalars = np.concatenate([base, ctx_oh])
    slot_ids = []; slot_sc = []
    for pkm in [my_a] + (me.get("bench") or [])[:N_BENCH] + [op_a] + (opp.get("bench") or [])[:N_BENCH]:
        slot_ids.append(pkm.get("id", 0) if isinstance(pkm, dict) else 0); slot_sc.extend(_slot_sc(pkm))
    while len(slot_ids) < N_SLOTS: slot_ids.append(0); slot_sc.extend([0, 0, 0, 0])
    slot_ids = np.array(slot_ids[:N_SLOTS], dtype=np.int64)
    slot_sc = np.array(slot_sc[:N_SLOTS * 4], dtype=np.float32)
    hand_ids = [_cid(c) for c in (me.get("hand") or [])][:N_HAND_PAD]
    while len(hand_ids) < N_HAND_PAD: hand_ids.append(0)
    hand_ids = np.array(hand_ids, dtype=np.int64)
    stadium = _cid(cur.get("stadium"))
    n = len(options)
    o_type = np.zeros(n, dtype=np.int64); o_area = np.zeros(n, dtype=np.int64)
    o_card = np.zeros(n, dtype=np.int64); o_tgt = np.zeros(n, dtype=np.int64)
    o_atk = np.zeros(n, dtype=np.int64); o_sc = np.zeros((n, 2), dtype=np.float32)
    for i, opt in enumerate(options):
        o_type[i] = opt.get("type", 0); o_area[i] = opt.get("area", 0)
        cid, tid, aid = _resolve(obs, pi, opt)
        o_card[i] = cid; o_tgt[i] = tid; o_atk[i] = aid
        o_sc[i, 0] = opt.get("count", 0); o_sc[i, 1] = opt.get("inPlayIndex", 0)
    return state_scalars, slot_ids, slot_sc, hand_ids, stadium, o_type, o_area, o_card, o_tgt, o_atk, o_sc, sel.get("maxCount", 1) or 1

def _forward(enc):
    state_scalars, slot_ids, slot_sc, hand_ids, stadium, o_type, o_area, o_card, o_tgt, o_atk, o_sc, max_count = enc
    slot = W_card[slot_ids].reshape(-1)                 # 768
    hand = W_card[hand_ids].mean(axis=0)                # 64
    stad = W_card[stadium]                              # 64
    s = np.concatenate([state_scalars, slot, slot_sc, hand, stad]).astype(np.float32)
    sv = _relu(_relu(s @ S0W.T + S0B) @ S2W.T + S2B)    # 256
    # options (n, ...)
    n = len(o_type)
    oh_t = np.eye(17, dtype=np.float32)[np.clip(o_type, 0, 16)]
    oh_a = np.eye(12, dtype=np.float32)[np.clip(o_area, 0, 11)]
    o = np.concatenate([oh_t, oh_a, W_card[o_card], W_card[o_tgt], W_atk[o_atk], o_sc], axis=1)  # n,223
    ov = _relu(_relu(o @ O0W.T + O0B) @ O2W.T + O2B)    # n,128
    sv_e = np.broadcast_to(sv, (n, sv.shape[0]))
    sc_in = np.concatenate([sv_e, ov], axis=1)          # n,384
    sc = _relu(sc_in @ C0W.T + C0B) @ C2W.T + C2B       # n,1
    return sc[:, 0], max_count

def _legal_fallback(obs_dict, sel):
    n = len(sel.get("option", []))
    k = sel.get("maxCount", 1) or 1
    return list(range(min(k, n)))

def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            return my_deck
        enc = _encode(obs_dict)
        if enc is None:
            return _legal_fallback(obs_dict, obs_dict["select"])
        scores, k = _forward(enc)
        k = max(1, min(int(k), len(scores)))
        # top-k by score (descending), tie-break by index for determinism
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return order[:k]
    except Exception:
        import traceback; traceback.print_exc()
        try:
            sel = obs_dict.get("select")
            if sel is None:
                return my_deck
            return _legal_fallback(obs_dict, sel)
        except Exception:
            traceback.print_exc()
            return [0]
