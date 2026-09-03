#!/usr/bin/env python3
"""Phase 2C.2 — Label-Preservation Audit.

Implements docs/phase2c2_label_preservation_audit_protocol.md EXACTLY.
Validation split only (never test). No training, no checkpoint access.

Pass 1 (default): render the frozen augmented-view sample, recover the
actual kornia transform parameters per view, compute the automated
structural proxy + per-dataset "plausibly preserved" verdict, write the
blinded human annotation sheet, and write a partial summary.json
(human-dependent quantities marked pending).

Pass 2 (--with-human): read artifacts/label_preservation_audit/human_scores.csv
(columns: item_id,score with score in {0,1,2}), compute p_gone, the
proxy-human agreement, and the final per-dataset verdict per protocol
sec. 7. Rewrites summary.json.

Usage:
    uv run python scripts/label_preservation_audit.py
    uv run python scripts/label_preservation_audit.py --with-human
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from when_tta_hurts.data import load_dataset  # noqa: E402
from when_tta_hurts.evaluation.views import stable_view_seed  # noqa: E402
from when_tta_hurts.transforms.policies import build_policy  # noqa: E402

AUDIT_SALT = "phase2c2_label_preservation"
DATASETS = ("pathmnist", "bloodmnist", "dermamnist")
RES = 28
N_IMG = 200
N_VIEW = 16
N_HUMAN = 50  # per dataset
OUT = REPO / "artifacts" / "label_preservation_audit"
PROTOCOL = REPO / "docs" / "phase2c2_label_preservation_audit_protocol.md"
VAL_CFG = REPO / "configs" / "validation_evaluation.yaml"

# frozen per-dataset decision-rule thresholds (protocol sec. 4)
FG_MIN = {"bloodmnist": 0.60}
CENTER_MIN = {"dermamnist": 0.70}
CROP_AREA_MIN = 0.50
INTENSITY_SHIFT_MAX = 0.45
# frozen verdict thresholds (protocol sec. 7)
P_GONE_MATERIAL = 0.15
A_NOTPRESERVED_MATERIAL = 0.25


def seed_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def load_tta_seed() -> int:
    cfg = yaml.safe_load(VAL_CFG.read_text())
    return int(cfg["confirmatory_tta_seed"])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def recover_params(policy: torch.nn.Sequential) -> dict:
    """Pull the actually-sampled parameters out of each kornia op's
    ._params after one forward pass. Op order (frozen): hflip, vflip,
    rotation, resized-crop, colorjitter, gaussian-blur."""
    hflip, vflip, rot, rrc, cj, gb = policy

    def applied(op) -> bool:
        bp = op._params["batch_prob"]
        return bool(torch.as_tensor(bp).reshape(-1)[0].item())

    src = rrc._params["src"].float().reshape(4, 2).cpu().numpy()  # 4 corners (x, y)
    x0, y0 = src[:, 0].min(), src[:, 1].min()
    x1, y1 = src[:, 0].max(), src[:, 1].max()
    return {
        "hflip": applied(hflip),
        "vflip": applied(vflip),
        "angle_deg": float(torch.as_tensor(rot._params["degrees"]).reshape(-1)[0].item())
        if applied(rot)
        else 0.0,
        "crop_x0": float(x0), "crop_y0": float(y0), "crop_x1": float(x1), "crop_y1": float(y1),
        "brightness_factor": float(torch.as_tensor(cj._params["brightness_factor"]).reshape(-1)[0].item()),
        "contrast_factor": float(torch.as_tensor(cj._params["contrast_factor"]).reshape(-1)[0].item()),
        "blur_sigma": float(torch.as_tensor(gb._params["sigma"]).reshape(-1)[0].item())
        if applied(gb)
        else 0.0,
    }


def _fwd_geom(xy: np.ndarray, p: dict) -> np.ndarray:
    """Map original-image pixel coords through hflip -> vflip -> rotation
    (about image center), matching the frozen policy op order, into the
    pre-crop frame. xy: (K, 2) array of (x, y)."""
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    c = (RES - 1) / 2.0
    if p["hflip"]:
        x = (RES - 1) - x
    if p["vflip"]:
        y = (RES - 1) - y
    th = np.deg2rad(p["angle_deg"])
    ct, st = np.cos(th), np.sin(th)
    xr = c + ct * (x - c) - st * (y - c)
    yr = c + st * (x - c) + ct * (y - c)
    return np.stack([xr, yr], axis=1)


def _inside_crop(xy: np.ndarray, p: dict) -> np.ndarray:
    return (
        (xy[:, 0] >= p["crop_x0"]) & (xy[:, 0] <= p["crop_x1"])
        & (xy[:, 1] >= p["crop_y0"]) & (xy[:, 1] <= p["crop_y1"])
    )


def crop_area_retained(p: dict) -> float:
    return float(((p["crop_x1"] - p["crop_x0"]) * (p["crop_y1"] - p["crop_y0"])) / (RES * RES))


def center_retained(p: dict) -> float:
    lo, hi = 0.2 * RES, 0.8 * RES
    g = np.linspace(lo, hi, 50)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return float(_inside_crop(_fwd_geom(pts, p), p).mean())


def foreground_retained(mask: np.ndarray, p: dict) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 1.0
    pts = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    return float(_inside_crop(_fwd_geom(pts, p), p).mean())


def foreground_mask(clean_rgb01: np.ndarray) -> np.ndarray:
    """Otsu threshold on the HSV saturation channel; foreground = the
    higher-saturation side (the stained cell / the lesion vs. pale bg)."""
    r, g, b = clean_rgb01[..., 0], clean_rgb01[..., 1], clean_rgb01[..., 2]
    mx = clean_rgb01.max(-1)
    mn = clean_rgb01.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-8, None), 0.0)
    vals = (sat * 255).astype(np.uint8).ravel()
    hist = np.bincount(vals, minlength=256).astype(float)
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1 + 1e-12)
    t = int(np.nanargmax(between))
    return sat * 255 > t


def is_preserved(dataset: str, m: dict) -> bool:
    if m["crop_area_retained"] < CROP_AREA_MIN:
        return False
    if m["intensity_shift"] > INTENSITY_SHIFT_MAX:
        return False
    if dataset == "bloodmnist" and m["foreground_retained"] < FG_MIN["bloodmnist"]:
        return False
    if dataset == "dermamnist" and m["center_retained"] < CENTER_MIN["dermamnist"]:
        return False
    return True


# ---------------------------------------------------------------------------
def render_png(path: Path, rgb01: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8)).save(path)


def pass1() -> None:
    tta_seed = load_tta_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "views").mkdir(exist_ok=True)

    per_view_rows: list[dict] = []
    auto_summary: dict = {}
    human_items: list[dict] = []

    for d in DATASETS:
        ds = load_dataset(d, split="val", size=RES, root=str(REPO / "data" / "raw"), download=True)
        imgs = np.asarray(ds.imgs)  # (N, 28, 28, 3) uint8
        labels = np.asarray(ds.labels).reshape(-1)
        classes = list(load_json_info(d))
        rng = np.random.default_rng(seed_int(f"{AUDIT_SALT}|images|{d}|{tta_seed}"))
        idxs = np.sort(rng.choice(len(imgs), size=N_IMG, replace=False))

        pol_cache = build_policy("mixed", (RES, RES))
        for si in idxs:
            clean01 = imgs[si].astype(np.float32) / 255.0
            mask = foreground_mask(clean01) if d in ("bloodmnist", "dermamnist") else None
            x = torch.from_numpy(clean01).permute(2, 0, 1).unsqueeze(0)
            for v in range(N_VIEW):
                torch.manual_seed(stable_view_seed(tta_seed, d, RES, int(si), v))
                with torch.no_grad():
                    _ = pol_cache(x)
                p = recover_params(pol_cache)
                m = {
                    "crop_area_retained": crop_area_retained(p),
                    "center_retained": center_retained(p),
                    "foreground_retained": foreground_retained(mask, p) if mask is not None else None,
                    "intensity_shift": abs(1.0 - p["brightness_factor"]) + abs(1.0 - p["contrast_factor"]),
                    "blur_sigma": p["blur_sigma"],
                }
                preserved = is_preserved(d, {**m, "foreground_retained": m["foreground_retained"] or 1.0})
                per_view_rows.append(
                    {"dataset": d, "sample_index": int(si), "view_index": v,
                     "class": classes[int(labels[si])] if classes else int(labels[si]),
                     **{k: p[k] for k in p}, **m, "automated_preserved": preserved}
                )

        sub = [r for r in per_view_rows if r["dataset"] == d]
        n_not = sum(1 for r in sub if not r["automated_preserved"])
        auto_summary[d] = {
            "n_views": len(sub),
            "a_not_preserved": n_not / len(sub),
            "crop_area_retained": pctl([r["crop_area_retained"] for r in sub]),
            "center_retained": pctl([r["center_retained"] for r in sub]),
            "foreground_retained": pctl([r["foreground_retained"] for r in sub if r["foreground_retained"] is not None])
            if d in ("bloodmnist", "dermamnist") else None,
            "intensity_shift": pctl([r["intensity_shift"] for r in sub]),
            "blur_sigma": pctl([r["blur_sigma"] for r in sub]),
        }

        # frozen stratified human sample: 25 preserved + 25 not-preserved
        hr = np.random.default_rng(seed_int(f"{AUDIT_SALT}|human|{d}|{tta_seed}"))
        pres = [r for r in sub if r["automated_preserved"]]
        notp = [r for r in sub if not r["automated_preserved"]]
        take_notp = min(N_HUMAN // 2, len(notp))
        take_pres = N_HUMAN - take_notp
        chosen = (
            [notp[i] for i in hr.choice(len(notp), size=take_notp, replace=False)]
            + [pres[i] for i in hr.choice(len(pres), size=min(take_pres, len(pres)), replace=False)]
        )
        for r in chosen:
            si, v = r["sample_index"], r["view_index"]
            clean01 = imgs[si].astype(np.float32) / 255.0
            x = torch.from_numpy(clean01).permute(2, 0, 1).unsqueeze(0)
            torch.manual_seed(stable_view_seed(tta_seed, d, RES, int(si), v))
            with torch.no_grad():
                aug = pol_cache(x)[0].permute(1, 2, 0).cpu().numpy()
            item_id = f"{d}-{si}-{v}"
            render_png(OUT / "views" / f"{item_id}__clean.png", clean01)
            render_png(OUT / "views" / f"{item_id}__aug.png", aug)
            human_items.append(
                {"item_id": item_id, "dataset": d, "class": r["class"],
                 "clean_png": f"views/{item_id}__clean.png", "aug_png": f"views/{item_id}__aug.png"}
            )

    # frozen shuffled annotation order (dataset identity not grouped)
    order_rng = np.random.default_rng(seed_int(f"{AUDIT_SALT}|order|{load_tta_seed()}"))
    order = order_rng.permutation(len(human_items))
    human_items = [human_items[i] for i in order]

    write_csv(OUT / "per_view.csv", per_view_rows)
    write_csv(OUT / "human_sheet.csv", human_items)
    with (OUT / "human_scores_TEMPLATE.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "score"])
        for it in human_items:
            w.writerow([it["item_id"], ""])

    summary = {
        "phase": "2C.2",
        "protocol_sha256": sha256_file(PROTOCOL),
        "confirmatory_tta_seed": load_tta_seed(),
        "n_img_per_dataset": N_IMG, "n_view_per_image": N_VIEW, "n_human_per_dataset": N_HUMAN,
        "frozen_thresholds": {
            "crop_area_min": CROP_AREA_MIN, "intensity_shift_max": INTENSITY_SHIFT_MAX,
            "foreground_min": FG_MIN, "center_min": CENTER_MIN,
            "p_gone_material": P_GONE_MATERIAL, "a_not_preserved_material": A_NOTPRESERVED_MATERIAL,
        },
        "automated": auto_summary,
        "human": "pending -- fill artifacts/label_preservation_audit/human_scores.csv, rerun with --with-human",
        "verdict": "pending",
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "manifest.json", {
        "protocol_sha256": sha256_file(PROTOCOL),
        "per_view_csv_sha256": sha256_file(OUT / "per_view.csv"),
        "human_sheet_csv_sha256": sha256_file(OUT / "human_sheet.csv"),
        "confirmatory_tta_seed": load_tta_seed(),
    })
    print(f"pass 1 done. {len(per_view_rows)} views. automated not-preserved rate:")
    for d in DATASETS:
        print(f"  {d:12} {auto_summary[d]['a_not_preserved']*100:5.1f}%   "
              f"(crop_area p50 {auto_summary[d]['crop_area_retained']['p50']:.2f}, "
              f"intensity_shift p95 {auto_summary[d]['intensity_shift']['p95']:.2f})")
    print(f"\nnow annotate: copy human_scores_TEMPLATE.csv -> human_scores.csv, fill score in {{0,1,2}},")
    print("then: uv run python scripts/label_preservation_audit.py --with-human")


def pass2_with_human() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    scores_path = OUT / "human_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found. Fill it from human_scores_TEMPLATE.csv first.")
    scores: dict[str, int] = {}
    with scores_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("score", "").strip() == "":
                raise SystemExit(f"blank score for {row['item_id']!r}; every item must be scored 0/1/2.")
            scores[row["item_id"]] = int(row["score"])

    per_view = list(csv.DictReader((OUT / "per_view.csv").open()))
    by_id = {f"{r['dataset']}-{r['sample_index']}-{r['view_index']}": r for r in per_view}

    human_block: dict = {}
    verdict: dict = {}
    for d in DATASETS:
        ids = [i for i in scores if i.startswith(d + "-")]
        s = np.array([scores[i] for i in ids])
        auto_pres = np.array([by_id[i]["automated_preserved"] in ("True", "true", True) for i in ids])
        human_present = s >= 1
        agree = float((human_present == auto_pres).mean())
        # Cohen's kappa
        po = agree
        pe = (human_present.mean() * auto_pres.mean()
              + (1 - human_present.mean()) * (1 - auto_pres.mean()))
        kappa = float((po - pe) / (1 - pe)) if pe < 1 else 1.0
        p_gone = float((s == 0).mean())
        human_block[d] = {
            "n_scored": len(ids),
            "score0_content_gone": p_gone,
            "score1_partial": float((s == 1).mean()),
            "score2_clear": float((s == 2).mean()),
            "proxy_human_raw_agreement": agree,
            "proxy_human_cohen_kappa": kappa,
            "human_score_dist_when_auto_preserved": _dist([sc for sc, ap in zip(s, auto_pres) if ap]),
            "human_score_dist_when_auto_not_preserved": _dist([sc for sc, ap in zip(s, auto_pres) if not ap]),
        }
        a_not = summary["automated"][d]["a_not_preserved"]
        material = (p_gone >= P_GONE_MATERIAL) or (a_not >= A_NOTPRESERVED_MATERIAL)
        verdict[d] = {
            "rule": "A_confound_material" if material else "B_confound_unlikely_dominant",
            "p_gone": p_gone, "a_not_preserved": a_not,
            "manuscript_action": (
                "report both rates; state augmentation-severity / label non-preservation cannot be "
                "excluded as a substantial contributor to this dataset's harm; soften causal language; "
                "list as a primary limitation."
                if material else
                "report both rates; state label non-preservation affects only a minority of views and is "
                "unlikely to be the dominant driver of this dataset's >=40pp harm; disclose residual uncertainty."
            ),
        }

    summary["human"] = human_block
    summary["verdict"] = verdict
    summary["human_scores_csv_sha256"] = sha256_file(scores_path)
    summary["annotation"] = "single annotator (frozen at protocol time); no inter-annotator kappa."
    write_json(OUT / "summary.json", summary)
    print("pass 2 done. per-dataset verdict:")
    for d in DATASETS:
        print(f"  {d:12} {verdict[d]['rule']:32}  p_gone={verdict[d]['p_gone']*100:.1f}%  "
              f"a_not_preserved={verdict[d]['a_not_preserved']*100:.1f}%  "
              f"(proxy-human kappa {human_block[d]['proxy_human_cohen_kappa']:.2f})")


# ---------------------------------------------------------------------------
def _dist(vals) -> dict:
    a = np.array(vals)
    return {str(k): float((a == k).mean()) if a.size else None for k in (0, 1, 2)}


def pctl(vals) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {"mean": float(a.mean()), "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json_info(dataset: str) -> list[str]:
    try:
        from medmnist import INFO

        return list(INFO[dataset]["label"].values())
    except Exception:
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-human", action="store_true")
    args = ap.parse_args()
    if args.with_human:
        pass2_with_human()
    else:
        pass1()
