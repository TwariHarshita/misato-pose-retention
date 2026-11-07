"""Standalone full-dataset extraction for the pose-retention project.

Self-contained on purpose: no imports from any agent workspace, so it keeps
running after the session that launched it has ended. Needs only requests,
h5py and numpy.

    python run_full_extraction.py

Resumes automatically. Safe to stop and restart at any time; at most the
complexes since the last checkpoint are repeated.

Design constraints, both measured rather than assumed:
  * Zenodo serialises the client, so threads do not help. Six workers were
    measured slower than one. This runs serially.
  * Zenodo rate-limits anonymous clients at roughly one request per second.
    Without pacing, a sustained run earns HTTP 429 within a few hundred
    requests; an earlier attempt without retry wrongly blacklisted 547
    complexes as bad when they were merely throttled.
"""
import io
import json
import os
import time
from collections import OrderedDict

import h5py
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://zenodo.org/records/7711953/files/"
URL = BASE + "MD.hdf5"
FIELDS = ["frames_rmsd_ligand", "frames_bSASA", "frames_distance",
          "frames_interaction_energy"]
NFRAMES = 100
CHECKPOINT = 250
MIN_INTERVAL = 1.05
MAX_RETRIES = 8
OUT = os.path.join(HERE, "misato_full_traces.npz")
STATE = os.path.join(HERE, "misato_full_state.json")
LOG = os.path.join(HERE, "extraction.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class RangeFile(io.RawIOBase):
    """Seekable file over HTTP byte ranges, block cached, paced and retrying."""

    def __init__(self, url, block=1 << 16, max_blocks=200000):
        self.block = block
        self.max_blocks = max_blocks
        self.s = requests.Session()
        h = self.s.head(url, allow_redirects=True, timeout=120)
        h.raise_for_status()
        self.url = h.url
        self.size = int(h.headers["Content-Length"])
        self.pos = 0
        self.cache = OrderedDict()
        self.n_req = 0
        self.n_retry = 0
        self.n_429 = 0
        self._next_at = 0.0

    def _throttle(self):
        wait = self._next_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next_at = time.monotonic() + MIN_INTERVAL

    def _get(self, idx):
        d = self.cache.get(idx)
        if d is not None:
            self.cache.move_to_end(idx)
            return d
        start = idx * self.block
        end = min(start + self.block, self.size) - 1
        data = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                               timeout=180)
            except Exception:
                self.n_retry += 1
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                self.n_retry += 1
                self.n_429 += (r.status_code == 429)
                ra = r.headers.get("Retry-After")
                delay = float(ra) if (ra or "").replace(".", "", 1).isdigit() \
                    else min(120, 5 * 2 ** attempt)
                time.sleep(delay)
                continue
            r.raise_for_status()
            data = r.content
            break
        if data is None:
            raise IOError(f"block {idx} failed after {MAX_RETRIES} attempts")
        self.n_req += 1
        self.cache[idx] = data
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx = self.pos // self.block
            off = self.pos - idx * self.block
            chunk = self._get(idx)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def all_ids():
    ids = []
    for name in ["train_MD.txt", "val_MD.txt", "test_MD.txt"]:
        r = requests.get(BASE + name, timeout=300)
        r.raise_for_status()
        ids += [x.strip() for x in r.text.split() if x.strip()]
    return ids


def _replace_with_retry(src, dst, tries=30, delay=20):
    """os.replace, tolerant of a reader briefly holding the destination open.

    On Windows a file with an open handle cannot be replaced, and anything
    inspecting progress while the run proceeds (np.load leaves the archive
    open until closed) will hold one. A transient reader must never end a
    multi-hour run, so wait it out rather than propagating the error.
    """
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == 0:
                log(f"  {os.path.basename(dst)} is locked by another process, waiting")
            time.sleep(delay)
    raise IOError(f"{dst} stayed locked for {tries * delay}s")


def save(done, store, bad):
    tmp = OUT + ".tmp.npz"
    np.savez_compressed(tmp, ids=np.array(done, dtype=object),
                        **{f: np.asarray(store[f], dtype=np.float32) for f in FIELDS})
    _replace_with_retry(tmp, OUT)
    with open(STATE + ".tmp", "w") as f:
        json.dump({"done": done, "bad": bad}, f)
    _replace_with_retry(STATE + ".tmp", STATE)


def main():
    log(f"starting, pid {os.getpid()}, output {HERE}")
    ids = all_ids()
    done, bad, store = [], [], {f: [] for f in FIELDS}
    if os.path.exists(OUT) and os.path.exists(STATE):
        with open(STATE) as f:
            st = json.load(f)
        done = st["done"]
        bad = [b for b in st["bad"] if "frame count" in b[1]]
        requeued = len(st["bad"]) - len(bad)
        with np.load(OUT, allow_pickle=True) as z:
            store = {f: list(z[f]) for f in FIELDS}
        log(f"resumed with {len(done)} stored, {requeued} transient re-queued, "
            f"{len(bad)} genuine")
    have = set(done) | {b[0] for b in bad}
    todo = [p for p in ids if p not in have]
    log(f"total {len(ids)}, todo {len(todo)}")

    fh = RangeFile(URL)
    h5 = h5py.File(fh, "r")
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        try:
            g = h5[p]
            vals = {f: np.asarray(g[f][:], dtype=np.float32) for f in FIELDS}
            if any(v.shape[0] != NFRAMES for v in vals.values()):
                bad.append((p, "frame count "
                            + str({f: int(v.shape[0]) for f, v in vals.items()})))
            else:
                done.append(p)
                for f in FIELDS:
                    store[f].append(vals[f])
        except Exception as e:
            bad.append((p, f"{type(e).__name__}: {e}"[:150]))
        if i % CHECKPOINT == 0:
            save(done, store, bad)
            el = time.time() - t0
            log(f"{i}/{len(todo)}  stored={len(done)}  bad={len(bad)}  "
                f"{el/3600:.2f}h  {el/i:.2f}s/complex  "
                f"eta {(len(todo)-i)*el/i/3600:.1f}h  "
                f"retries={fh.n_retry} rate_limited={fh.n_429}")
    save(done, store, bad)
    log(f"FINISHED stored={len(done)} bad={len(bad)} {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ABORTED {type(exc).__name__}: {exc}")
        raise
