"""Quality-control classification for MISATO ligand-retention traces.

Extreme values are not discarded silently. Each complex is assigned to exactly
one class; every reported distribution states how many complexes fell into each
class, and only ANALYSED complexes enter summary statistics.

Classes
-------
ANALYSED            normal behaviour, enters the reference distribution
UNBOUND             ligand has left the protein and is free in solvent
DISPLACED_AT_START  already far from the native pose in the first frames
IMAGING_SUSPECT     far from the native pose yet still substantially buried,
                    which is self-contradictory under a single periodic image

Rationale for the thresholds
----------------------------
UNBOUND uses buried SASA rather than RMSD, because an RMSD large enough to
imply unbinding can also be produced by periodic-boundary imaging; zero buried
surface is the physical statement.

DISPLACED_AT_START exists because MD here starts from the native structure, so
the first frames can only show relaxation. A ligand already several angstrom
away at 0.5 ns has not relaxed; the usual causes are a crystallisation additive
being tracked instead of the real ligand, a failed setup, or imaging. These
need inspection, not averaging.

IMAGING_SUSPECT catches the combination that no single consistent image can
produce: a ligand reported tens of angstrom from its native pose while still
burying hundreds of square angstrom of surface.
"""
import numpy as np

ANALYSED = "ANALYSED"
UNBOUND = "UNBOUND"
DISPLACED_AT_START = "DISPLACED_AT_START"
IMAGING_SUSPECT = "IMAGING_SUSPECT"

NON_PHYSICAL = "NON_PHYSICAL"

ORDER = [ANALYSED, UNBOUND, DISPLACED_AT_START, IMAGING_SUSPECT, NON_PHYSICAL]


def classify(rmsd, bsasa, start_frames=5, end_frames=5,
             start_rmsd_max=5.0, imaging_rmsd=15.0, imaging_bsasa=50.0):
    """Classify one complex from its per-frame ligand RMSD and buried SASA.

    Parameters
    ----------
    rmsd, bsasa : 1-D arrays of equal length, one value per frame.
    start_rmsd_max : angstrom. Above this in the opening frames the complex is
        DISPLACED_AT_START.
    imaging_rmsd, imaging_bsasa : a frame exceeding `imaging_rmsd` while burying
        more than `imaging_bsasa` marks the complex IMAGING_SUSPECT.

    Returns
    -------
    dict with keys: qc_class, rmsd_start, rmsd_end, bsasa_start, bsasa_end,
    rmsd_max, n_frames.
    """
    rmsd = np.asarray(rmsd, dtype=float)
    bsasa = np.asarray(bsasa, dtype=float)
    rec = {
        "n_frames": int(rmsd.size),
        "rmsd_start": float(rmsd[:start_frames].mean()),
        "rmsd_end": float(rmsd[-end_frames:].mean()),
        "rmsd_max": float(rmsd.max()),
        "bsasa_start": float(bsasa[:start_frames].mean()),
        "bsasa_end": float(bsasa[-end_frames:].mean()),
    }
    contradictory = np.any((rmsd > imaging_rmsd) & (bsasa > imaging_bsasa))
    # Buried surface area cannot be negative. A negative value means the upstream
    # calculation failed for that complex, so nothing derived from it is
    # trustworthy, including its RMSD. This is checked first, before any class
    # that reads a SASA value.
    if np.any(bsasa < 0.0):
        rec["qc_class"] = NON_PHYSICAL
    elif rec["rmsd_start"] > start_rmsd_max:
        rec["qc_class"] = DISPLACED_AT_START
    elif contradictory:
        rec["qc_class"] = IMAGING_SUSPECT
    elif rec["bsasa_end"] == 0.0:
        rec["qc_class"] = UNBOUND
    else:
        rec["qc_class"] = ANALYSED
    return rec


def classify_npz(z, pdb_ids, **kw):
    """Apply `classify` to every id present in a loaded npz of traces."""
    out = []
    for p in pdb_ids:
        key_r, key_b = f"{p}|frames_rmsd_ligand", f"{p}|frames_bSASA"
        if key_r not in z or key_b not in z:
            continue
        rec = classify(z[key_r], z[key_b], **kw)
        rec["pdb_id"] = p
        out.append(rec)
    return out


def summarise(records):
    """Counts per class, in a fixed order, for reporting alongside any figure."""
    from collections import Counter
    c = Counter(r["qc_class"] for r in records)
    return {k: int(c.get(k, 0)) for k in ORDER}
