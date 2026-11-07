"""Reproduce every number in the pose-retention manuscript from the extracted traces.

Inputs (all in the same directory as this script)
    misato_full_traces.npz   per-frame ligand RMSD and buried SASA for 16,971 complexes,
                             produced by run_full_extraction.py from MISATO MD.hdf5
    ligand_counts_full.csv   ligand atom and heavy-atom counts, produced by
                             run_topology_pass.py from the same file (its output
                             ligand_counts.csv is used if the _full file is absent)
    QM.hdf5                  optional, MISATO semi-empirical ligand properties; needed only
                             for the descriptor analysis (Figure 5)
    protein_groups.csv       optional, pdb_id -> UniProt accession, for protein-grouped CV

Usage
    python reproduce_analysis.py            # everything except the boosted models (~1 min)
    python reproduce_analysis.py --models   # also the quantile-model gains (~15 min)

Outputs
    numbers.json             every quantity quoted in the manuscript
    per_complex.csv          one row per complex: QC class, displacement, burial, enclosure

Conventions, all as stated in the Methods
    displacement at time t   mean ligand RMSD over the five frames ending at t (0.5 ns)
    10 ns outcome            mean of frames 96-100; 5 ns window frames 46-55; 1 ns frames 1-10
    burial retained          mean buried SASA of the last five frames / first five frames
    enclosure                mean buried SASA of the first five frames / ligand heavy atoms
    confidence intervals     percentile bootstrap, 4,000 resamples, seed 2026
"""
import argparse
import json
import os

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retention_qc  # noqa: E402

B = 4000
SEED = 2026


def boot_quantile(v, q, rng, chunk=500):
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, len(v), (min(chunk, B - s), len(v)))
        out.append(np.quantile(v[idx], q, axis=1))
    lo, hi = np.percentile(np.concatenate(out), [2.5, 97.5])
    return [round(float(np.quantile(v, q)), 3), round(float(lo), 3), round(float(hi), 3)]


def boot_frac_above(v, thr, rng, chunk=500):
    out = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, len(v), (min(chunk, B - s), len(v)))
        out.append((v[idx] > thr).mean(1) * 100)
    lo, hi = np.percentile(np.concatenate(out), [2.5, 97.5])
    return [round(float((v > thr).mean() * 100), 2), round(float((hi - lo) / 2), 2)]


def pinball(y, p, a):
    d = y - p
    return np.mean(np.maximum(a * d, (a - 1) * d))


def main(run_models):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rng = np.random.default_rng(SEED)
    with np.load("misato_full_traces.npz", allow_pickle=True) as z:
        ids = [str(x) for x in z["ids"]]
        R = z["frames_rmsd_ligand"].astype(float)
        S = z["frames_bSASA"].astype(float)

    # ---------------------------------------------------------------- quality control
    qc = [retention_qc.classify(R[i], S[i])["qc_class"] for i in range(len(ids))]
    qc = np.array(qc)
    counts = {k: int((qc == k).sum()) for k in retention_qc.ORDER}
    ok = qc == retention_qc.ANALYSED
    N = {"n_pooled": len(ids), "n_analysed": int(ok.sum()), "qc": counts,
         "qc_fail_pct": round(100 * (1 - ok.mean()), 3)}

    # ---------------------------------------------------------------- reference distribution
    w10 = R[:, -5:].mean(1)
    w5 = R[:, 45:55].mean(1)
    w1 = R[:, :10].mean(1)
    v = w10[ok]
    for q in [10, 25, 50, 75, 90, 95]:
        N[f"p{q}"] = boot_quantile(v, q / 100, rng)
    for t in [2, 3, 5]:
        N[f"gt{t}"] = boot_frac_above(v, t, rng)

    # ---------------------------------------------------------------- convergence
    flip = (w5[ok] > 2) != (w10[ok] > 2)
    W = stats.wilcoxon(w10[ok], w5[ok])
    from sklearn.metrics import roc_auc_score
    N.update({"flip_n": int(flip.sum()), "flip_pct": round(float(100 * flip.mean()), 2),
              "drift_median": round(float(np.median(w10[ok] - w5[ok])), 3),
              "drift_p": float(W.pvalue),
              "drift_z": round(float(stats.norm.isf(W.pvalue / 2) * np.sign(np.median(w5[ok] - w10[ok]))), 1)
              if W.pvalue > 0 else None,
              "auc_1ns": round(float(roc_auc_score((v > 2).astype(int), w1[ok])), 3),
              "spearman_1ns_10ns": round(float(stats.spearmanr(w1[ok], v)[0]), 3)})

    # ---------------------------------------------------------------- burial retained
    s0 = S[:, :5].mean(1)
    s1 = S[:, -5:].mean(1)
    ret = s1 / np.where(s0 > 0, s0, np.nan)
    r_ok = ret[ok]
    above = v > 2
    good = ~np.isnan(r_ok)
    still = above & (r_ok >= 0.8) & good
    N["burial"] = {"n": int(good.sum()), "above2": int((above & good).sum()),
                   "still_buried": int(still.sum()),
                   "lost_burial": int((above & good & (r_ok < 0.8)).sum()),
                   "pct_of_rejected": round(float(100 * still.sum() / (above & good).sum()), 1),
                   "median_ret_above2": round(float(np.nanmedian(r_ok[above])), 3),
                   "median_ret_all": round(float(np.nanmedian(r_ok)), 3),
                   "spearman": round(float(stats.spearmanr(v[good], r_ok[good])[0]), 3)}

    # ---------------------------------------------------------------- robustness to QC
    inc = ok | (qc == retention_qc.NON_PHYSICAL)
    N["robustness_nonphysical"] = {
        "n_nonphys": counts[retention_qc.NON_PHYSICAL],
        "p50_with": round(float(np.median(w10[inc])), 2), "p50_without": round(float(np.median(v)), 2),
        "gt2_with": round(float(100 * (w10[inc] > 2).mean()), 1), "gt2_without": round(float(100 * (v > 2).mean()), 1),
        "p90_with": round(float(np.quantile(w10[inc], .9)), 2), "p90_without": round(float(np.quantile(v, .9)), 2)}

    # ---------------------------------------------------------------- enclosure
    per = pd.DataFrame({"pdb_id": ids, "qc_class": qc, "rmsd_1ns": w1, "rmsd_5ns": w5, "rmsd_10ns": w10,
                        "bsasa_start": s0, "bsasa_end": s1, "burial_retained": ret,
                        "changes_verdict_5_to_10ns": (w5 > 2) != (w10 > 2)})
    lc = pd.read_csv("ligand_counts_full.csv" if os.path.exists("ligand_counts_full.csv") else "ligand_counts.csv")
    per = per.merge(lc, on="pdb_id", how="left")
    per["enclosure"] = per.bsasa_start / per.n_heavy
    A = per[(per.qc_class == retention_qc.ANALYSED) & (per.n_heavy > 0) & (per.bsasa_start > 0)].copy()
    A["enclosure_quartile"] = pd.qcut(A.enclosure, 4, labels=[1, 2, 3, 4]).astype(int)
    per = per.merge(A[["pdb_id", "enclosure_quartile"]], on="pdb_id", how="left")
    strata = []
    for k, g in A.groupby("enclosure_quartile"):
        x = g.rmsd_10ns.values
        row = {"quartile": int(k), "n": len(g), "encl_lo": round(float(g.enclosure.min()), 1),
               "encl_hi": round(float(g.enclosure.max()), 1), "frac_gt2": round(float(100 * (x > 2).mean()), 1)}
        for q in [50, 75, 90, 95]:
            b = boot_quantile(x, q / 100, rng)
            row[f"p{q}"], row[f"p{q}_ci"] = b[0], b[1:]
        strata.append(row)
    rho = stats.spearmanr(A.enclosure, A.rmsd_10ns)[0]
    H, _ = stats.kruskal(*[g.rmsd_10ns.values for _, g in A.groupby("enclosure_quartile")])
    N["enclosure"] = {"n": len(A), "spearman": round(float(rho), 3),
                      "spearman_t": round(float(rho * np.sqrt((len(A) - 2) / (1 - rho ** 2))), 1),
                      "kruskal_H": round(float(H), 1),
                      "spearman_raw_bsasa": round(float(stats.spearmanr(A.bsasa_start, A.rmsd_10ns)[0]), 3),
                      "spearman_n_heavy": round(float(stats.spearmanr(A.n_heavy, A.rmsd_10ns)[0]), 3),
                      "strata": strata}
    # first frame versus 0.5 ns average
    e0 = S[[ids.index(p) for p in A.pdb_id], 0] / A.n_heavy.values
    rel = e0 / A.enclosure.values - 1
    q0 = pd.qcut(e0, 4, labels=False) + 1
    N["enclosure_first_frame"] = {
        "median_abs_rel_diff_pct": round(float(100 * np.median(np.abs(rel[e0 > 0]))), 1),
        "quartile_agreement_pct": round(float(100 * (q0 == A.enclosure_quartile.values).mean()), 1),
        "rank_corr": round(float(stats.spearmanr(e0, A.enclosure)[0]), 3)}

    # ---------------------------------------------------------------- descriptors and models
    if os.path.exists("QM.hdf5"):
        import h5py
        rows = []
        with h5py.File("QM.hdf5", "r") as qm:
            for p in A.pdb_id:
                if p not in qm:
                    continue
                mp, ap = qm[p]["mol_properties"], qm[p]["atom_properties"]
                nat, nb = len(ap["atom_names"]), len(ap["bonds"])
                rows.append({"pdb_id": p, "n_bonds": nb, "n_rings": nb - nat + 1,
                             "MW": float(mp["molecular_weight"][()]), "charge": float(mp["total_charge"][()]),
                             "hardness": float(mp["Hardness"][()]), "electroneg": float(mp["Electronegativity"][()]),
                             "ea": float(mp["Electron_Affinity"][()]), "ip": float(mp["Ionization_Potential"][()])})
        D = A.merge(pd.DataFrame(rows), on="pdb_id")
        lig = ["n_heavy", "n_bonds", "n_rings", "MW", "charge", "hardness", "electroneg", "ea", "ip"]
        N["descriptors"] = {"n": len(D), "spearman": {f: round(float(stats.spearmanr(D[f], D.rmsd_10ns)[0]), 3)
                                                      for f in lig + ["bsasa_start", "enclosure"]}}
        if run_models:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import GroupKFold
            grp = D.pdb_id.values
            if os.path.exists("protein_groups.csv"):
                g = pd.read_csv("protein_groups.csv").set_index("pdb_id").uniprot
                grp = D.pdb_id.map(g).fillna(D.pdb_id).values
            y = D.rmsd_10ns.values
            base = pinball(y, np.full(len(y), np.quantile(y, .9)), .9)
            sets = {"ligand only (no burial)": lig, "raw burial alone": ["bsasa_start"],
                    "enclosure alone": ["enclosure"], "enclosure + size": ["enclosure", "n_heavy", "n_bonds", "n_rings", "MW"],
                    "enclosure + electronic/charge": ["enclosure", "charge", "hardness", "electroneg", "ea", "ip"],
                    "ligand only + enclosure": lig + ["enclosure"]}
            gains = {}
            for name, cols in sets.items():
                X = D[cols].values
                pred = np.zeros(len(y))
                for tr, te in GroupKFold(5).split(X, y, grp):
                    pred[te] = GradientBoostingRegressor(loss="quantile", alpha=.9, n_estimators=150, max_depth=3,
                                                         learning_rate=.05, random_state=0).fit(X[tr], y[tr]).predict(X[te])
                gains[name] = {"protein_grouped_gain": round(float(100 * (1 - pinball(y, pred, .9) / base)), 1),
                               "grouped_coverage": round(float(100 * (y <= pred).mean()), 1)}
                print(f"  model {name}: {gains[name]}", flush=True)
            N["descriptors"]["gains"] = gains

    json.dump(N, open("numbers.json", "w"), indent=1)
    per.to_csv("per_complex.csv", index=False)
    print(f"done: {N['n_analysed']} analysed of {N['n_pooled']}; numbers.json and per_complex.csv written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="also fit the quantile models (slow)")
    main(ap.parse_args().models)
