#!/usr/bin/env python3
"""Train the IL move-policy net (CPU, PyTorch).

Architecture (shared card/attack embeddings, small MLP — numpy-freezable):
  state_vec  = MLP(state_scalars | slot_card_emb | slot_scalars | hand_emb.mean | stadium_emb)
  option_vec = MLP(type_oh | area_oh | card_emb | tgt_emb | atk_emb | opt_scalars)
  score      = MLP(state_vec | option_vec) -> 1   (per option)
  loss       = masked BCEWithLogits (picked=1)

Usage: python tools/il_train.py <decisions.pkl> <model.pt> [--epochs N] [--bs B]
"""
import sys, os, pickle, argparse, random
import numpy as np
import torch
import torch.nn as nn

N_CARDS = 1268          # card ids 1..1267 + 0 pad
N_ATTACKS = 1557        # 1..1556 + 0
N_BENCH = 5
N_SLOTS = 2 * (1 + N_BENCH)   # 12
N_HAND_PAD = 20
N_OPTION_SCALARS = 2
MAX_OPT = 64            # pad options per decision
CARD_DIM = 64
ATK_DIM = 64
HIDDEN = 256


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(N_CARDS, CARD_DIM, padding_idx=0)
        self.atk_emb = nn.Embedding(N_ATTACKS, ATK_DIM, padding_idx=0)
        # state: scalars(72) + slot_card(12*64=768) + slot_scalars(48) + hand_mean(64) + stadium(64) = 1016
        state_in = 72 + N_SLOTS * CARD_DIM + N_SLOTS * 4 + CARD_DIM + CARD_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        # option: type_oh(17) + area_oh(12) + card(64) + tgt(64) + atk(64) + scalars(2) = 223
        opt_in = 17 + 12 + CARD_DIM + CARD_DIM + ATK_DIM + N_OPTION_SCALARS
        self.opt_mlp = nn.Sequential(
            nn.Linear(opt_in, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(HIDDEN + HIDDEN // 2, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),
        )

    def forward(self, batch):
        B = batch["state_scalars"].shape[0]
        slot = self.card_emb(batch["slot_card_ids"]).reshape(B, -1)          # B, 12*64
        hand = self.card_emb(batch["hand_card_ids"]).mean(dim=1)             # B, 64
        stad = self.card_emb(batch["stadium_id"])                            # B, 64
        s = torch.cat([batch["state_scalars"], slot, batch["slot_scalars"], hand, stad], dim=1)
        state_vec = self.state_mlp(s)                                        # B, HIDDEN
        # options: (B, MAX_OPT, ...)
        n_opt = self.card_emb(batch["option_card_id"])                       # B, M, 64
        t_opt = self.card_emb(batch["option_target_id"])                     # B, M, 64
        a_opt = self.atk_emb(batch["option_attack_id"])                      # B, M, 64
        otype = nn.functional.one_hot(batch["option_type"].clamp(0, 16), 17).float()
        oarea = nn.functional.one_hot(batch["option_area"].clamp(0, 11), 12).float()
        o = torch.cat([otype, oarea, n_opt, t_opt, a_opt, batch["option_scalars"]], dim=-1)  # B,M,223
        opt_vec = self.opt_mlp(o)                                            # B, M, HIDDEN/2
        sv = state_vec.unsqueeze(1).expand(-1, opt_vec.size(1), -1)
        sc = self.score(torch.cat([sv, opt_vec], dim=-1)).squeeze(-1)        # B, M
        return sc


def collate(recs):
    def arr(name, dtype):
        return torch.from_numpy(np.stack([r[name] for r in recs]).astype(dtype))

    B = len(recs)
    state_scalars = arr("state_scalars", np.float32)
    slot_card_ids = arr("slot_card_ids", np.int64)
    slot_scalars = arr("slot_scalars", np.float32)
    hand_card_ids = arr("hand_card_ids", np.int64)
    stadium_id = torch.from_numpy(np.array([r["stadium_id"] for r in recs], dtype=np.int64))
    opt_type = torch.zeros(B, MAX_OPT, dtype=torch.int64)
    opt_area = torch.zeros(B, MAX_OPT, dtype=torch.int64)
    opt_card = torch.zeros(B, MAX_OPT, dtype=torch.int64)
    opt_tgt = torch.zeros(B, MAX_OPT, dtype=torch.int64)
    opt_atk = torch.zeros(B, MAX_OPT, dtype=torch.int64)
    opt_sc = torch.zeros(B, MAX_OPT, N_OPTION_SCALARS, dtype=torch.float32)
    picked = torch.zeros(B, MAX_OPT, dtype=torch.float32)
    mask = torch.zeros(B, MAX_OPT, dtype=torch.bool)
    max_count = torch.zeros(B, dtype=torch.float32)
    for i, r in enumerate(recs):
        n = min(len(r["option_type"]), MAX_OPT)
        opt_type[i, :n] = torch.from_numpy(r["option_type"][:n])
        opt_area[i, :n] = torch.from_numpy(r["option_area"][:n])
        opt_card[i, :n] = torch.from_numpy(r["option_card_id"][:n])
        opt_tgt[i, :n] = torch.from_numpy(r["option_target_id"][:n])
        opt_atk[i, :n] = torch.from_numpy(r["option_attack_id"][:n])
        opt_sc[i, :n] = torch.from_numpy(r["option_scalars"][:n])
        picked[i, :n] = torch.from_numpy(r["picked"][:n].astype(np.float32))
        mask[i, :n] = True
        max_count[i] = r["max_count"]
    return {
        "state_scalars": state_scalars, "slot_card_ids": slot_card_ids, "slot_scalars": slot_scalars,
        "hand_card_ids": hand_card_ids, "stadium_id": stadium_id,
        "option_type": opt_type, "option_area": opt_area, "option_card_id": opt_card,
        "option_target_id": opt_tgt, "option_attack_id": opt_atk, "option_scalars": opt_sc,
        "picked": picked, "mask": mask, "max_count": max_count,
    }


def topk_acc(model, recs):
    """Fraction of decisions where a picked option is in the model's top-maxCount."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(recs), 128):
            batch = collate(recs[i:i + 128])
            scores = model(batch)                        # B, M
            scores = scores.masked_fill(~batch["mask"], -1e9)
            mc = batch["max_count"].long().clamp(min=1)  # B
            for j in range(scores.size(0)):
                k = int(mc[j].item())
                topk = set(torch.topk(scores[j], k).indices.tolist())
                picked_idx = set((batch["picked"][j] == 1).nonzero().flatten().tolist())
                # only count within-mask picked
                picked_idx = {x for x in picked_idx if x < scores.size(1) and batch["mask"][j, x]}
                if picked_idx & topk:
                    correct += 1
    return correct / max(1, len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    recs = pickle.load(open(args.pkl, "rb"))
    print(f"loaded {len(recs)} decisions")
    recs = [r for r in recs if len(r["option_type"]) > 0]
    random.seed(0); random.shuffle(recs)
    n_val = max(1, len(recs) // 10)
    val, train = recs[:n_val], recs[n_val:]
    print(f"train={len(train)} val={len(val)}")

    model = PolicyNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(args.epochs):
        model.train()
        random.shuffle(train)
        tot = 0.0; nb = 0
        for i in range(0, len(train), args.bs):
            batch = collate(train[i:i + args.bs])
            scores = model(batch)
            # weight picked by 1/max_count so each decision contributes ~equally
            w = torch.where(batch["picked"] > 0, 1.0 / batch["max_count"].unsqueeze(1), torch.ones_like(batch["picked"]))
            loss = (bce(scores, batch["picked"]) * w * batch["mask"].float()).sum() / batch["mask"].float().sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        val_acc = topk_acc(model, val)
        print(f"epoch {ep+1}/{args.epochs}  train_loss={tot/nb:.4f}  val_topk_acc={val_acc:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"saved -> {args.out}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
