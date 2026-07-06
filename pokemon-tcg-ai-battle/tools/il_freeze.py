#!/usr/bin/env python3
"""Freeze a trained PolicyNet (model.pt) to a numpy .npz for pure-numpy deployment."""
import sys, os, argparse
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pt")
    ap.add_argument("out_npz")
    args = ap.parse_args()
    ckpt = torch.load(args.model_pt, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]
    arrs = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez(args.out_npz, **arrs)
    print(f"saved {len(arrs)} arrays -> {args.out_npz}")
    for k, v in arrs.items():
        print(f"  {k}: {v.shape}")

if __name__ == "__main__":
    main()
