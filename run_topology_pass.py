"""Second pass: ligand atom counts for every complex, for the enclosure variable.

The first pass (run_full_extraction.py) collected the per-frame traces. This one
collects what those traces cannot provide: the size of the ligand, needed to turn
buried surface area into buried surface area per heavy atom.

    python run_topology_pass.py

Resumable in the same way, checkpointing to CSV every 250 complexes.

Why two small datasets and not the whole topology: the ligand is the last entry
in molecules_begin_atom_index, so only the tail slice atoms_element[start:] is
read, typically tens of atoms rather than the several thousand in the full
system. Heavy atoms are those with element number other than 1.

Measured cost: about 5.5 s per complex, roughly 25 hours for the full set. This
is slower than the first pass because atoms_element is a far larger dataset than
the 100-point frame arrays, so more byte-range requests are needed per complex.

Validated against 40 complexes with independently known counts: n_atoms matches
exactly. Note that the older descriptor table had n_heavy equal to n_atoms, i.e.
hydrogens were never excluded; this script is the correction for that.
"""
import csv
import json
import os
import time

import h5py
import numpy as np

from run_full_extraction import RangeFile, URL, all_ids, log, _replace_with_retry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ligand_counts.csv")
CHECKPOINT = 250


def lig_counts(grp):
    """(n_atoms, n_heavy) for the ligand, reading only its tail of the element array."""
    beg = grp["molecules_begin_atom_index"][:]
    start = int(beg[-1])
    el = grp["atoms_element"][start:]
    return len(el), int((el != 1).sum())


def write(rows):
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "n_atoms", "n_heavy"])
        w.writerows(rows)
    _replace_with_retry(tmp, OUT)


def main():
    log(f"topology pass starting, pid {os.getpid()}")
    ids = all_ids()
    rows, done = [], set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([r["pdb_id"], int(r["n_atoms"]), int(r["n_heavy"])])
                done.add(r["pdb_id"])
        log(f"resumed with {len(done)} already counted")
    todo = [p for p in ids if p not in done]
    log(f"total {len(ids)}, todo {len(todo)}")

    fh = RangeFile(URL)
    h5 = h5py.File(fh, "r")
    t0 = time.time()
    failed = 0
    for i, p in enumerate(todo, 1):
        try:
            na, nh = lig_counts(h5[p])
            rows.append([p, na, nh])
        except Exception as e:
            failed += 1
            log(f"  {p} failed: {type(e).__name__}: {str(e)[:90]}")
        if i % CHECKPOINT == 0:
            write(rows)
            el = time.time() - t0
            log(f"{i}/{len(todo)}  counted={len(rows)}  failed={failed}  "
                f"{el/3600:.2f}h  {el/i:.2f}s/complex  "
                f"eta {(len(todo)-i)*el/i/3600:.1f}h  "
                f"retries={fh.n_retry} rate_limited={fh.n_429}")
    write(rows)
    log(f"FINISHED counted={len(rows)} failed={failed} {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ABORTED {type(exc).__name__}: {exc}")
        raise
