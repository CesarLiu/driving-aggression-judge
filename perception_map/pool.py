"""Pooling the tracks of every scene recorded at the same place, so that rebuilding the topology and
inferring the controls see more vehicles than one 9 s clip holds.

WOMD scenes share one coordinate frame per city, and scenes recorded at one place carry the same
painted lanes (point for point when cut from one drive, within centimetres on another day). ``extract`` stores each scene's vehicle tracks (one file per scene,
read on demand) and the start points of its painted lanes (rounded to SNAP m); ``index`` links two
scenes when SHARED or more of those points fall in the same GRID m cells, and lists for each scene the
others whose lanes come within NEAR m of its judged pair, those with most lanes there first. ``others_tracks`` reads up to MAX_OTHERS of them, dropping
copies of tracks already taken (scenes cut from one drive overlap in time); ``pooled`` appends them to
a scene as extra agents over the same steps. Their times differ from the scene's, which the rebuild
and counts of what vehicles did at a stop line do not care about; anything timed (light states, who
went while whom waited) must still come from the scene's own vehicles.

  python pool.py extract --scenes DIR --out pool --part 0/8
  python pool.py index --pool pool --pairs screen_crossing.csv
"""

import argparse
import copy
import glob
import pickle
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "judge-driving-aggression" / "scripts"
sys.path.insert(0, str(SKILL))
from scenes import load_scene  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from responsibility.lanes import LANE_TYPES  # noqa: E402

SNAP = 0.1  # m: lane start points rounded to this before comparing scenes
GRID = 1.0  # m: two scenes share a lane start when it falls in one cell of this grid in both
SHARED = 5  # shared painted lane starts that make two scenes one place
NEAR = 80.0  # m: a scene's tracks are pooled into another's when its lanes come this close to the judged pair
MAX_OTHERS = 50  # other scenes pooled into one at most
DUP_EVERY = 5  # steps between the track points compared when looking for copies
DUP_DIST = 0.05  # m: three sampled points this close to taken ones make a track a copy


def files_of(path, part):
    files = sorted(Path(p) for p in glob.glob(str(Path(path) / "*.pkl")))
    if part:
        k, n = (int(x) for x in part.split("/"))
        files = files[k::n]
    return files


def track_arrays(scene, keep):
    return {"ids": np.array([str(scene.track_ids[i]) for i in keep]),
            "position": scene.position[keep].astype(np.float32), "heading": scene.heading[keep].astype(np.float32),
            "velocity": scene.velocity[keep].astype(np.float32), "valid": scene.valid[keep],
            "size": np.stack([scene.length[keep], scene.width[keep], scene.height[keep]], -1).astype(np.float32)}


def cmd_extract(args):
    """Per scene: its vehicle tracks in tracks/<scene>.npz; per part: the painted lane starts."""
    folder = Path(args.out) / "tracks"
    folder.mkdir(parents=True, exist_ok=True)
    starts = {}
    for path in files_of(args.scenes, args.part):
        try:
            scene, _ = load_scene(path, Scene)
        except Exception as exc:
            print("skip %s: %s" % (path.stem, exc), file=sys.stderr)
            continue
        np.savez(folder / (path.stem + ".npz"),
                 **track_arrays(scene, [i for i in range(scene.n_agents) if scene.types[i] == "VEHICLE"]))
        st = [np.asarray(f["polyline"], float)[0, :2] for f in scene.map_features.values()
              if f.get("type") in LANE_TYPES and not f.get("interpolating") and len(f.get("polyline", [])) >= 2]
        starts[path.stem] = np.round(np.array(st).reshape(-1, 2) / SNAP).astype(np.int64)
    name = Path(args.out) / ("starts_%s.pkl" % (args.part.replace("/", "of") if args.part else "all"))
    pickle.dump(starts, open(name, "wb"), protocol=4)
    print("%d scenes -> %s" % (len(starts), name))


@lru_cache(maxsize=256)
def load_tracks(folder, name):
    with np.load(Path(folder) / "tracks" / (name + ".npz")) as z:
        return {k: z[k] for k in z.files}


def load_places(folder):
    return pickle.load(open(Path(folder) / "places.pkl", "rb"))


def cmd_index(args):
    """places.pkl: {scene: [other scenes of its place whose lanes reach its judged pair]}, and a summary."""
    import csv

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    starts = {}
    for p in sorted(glob.glob(str(Path(args.pool) / "starts_*.pkl"))):
        starts.update(pickle.load(open(p, "rb")))
    names = sorted(starts)
    cell = {n: np.floor(starts[n] * SNAP / GRID).astype(np.int64) for n in names}
    code = [np.unique(cell[n][:, 0] * (1 << 31) + cell[n][:, 1]) for n in names]
    _, col = np.unique(np.concatenate(code), return_inverse=True)
    row = np.repeat(np.arange(len(names)), [len(c) for c in code])
    m = coo_matrix((np.ones(len(col)), (row, col))).tocsr()
    shared = (m @ m.T).tocoo()  # painted lane starts two scenes have in common
    keep = (shared.data >= SHARED) & (shared.row != shared.col)
    _, place = connected_components(coo_matrix((shared.data[keep], (shared.row[keep], shared.col[keep])),
                                               shape=(len(names),) * 2), directed=False)
    members = defaultdict(list)
    for k, n in enumerate(names):
        members[place[k]].append(n)
    with open(args.pairs, encoding="utf-8") as f:
        pairs = {r["scene"]: r for r in csv.DictReader(f)}
    near = {}
    for k, n in enumerate(names):
        row = pairs.get(n)
        if row is None:
            continue
        t = load_tracks(args.pool, n)
        ids = list(t["ids"])
        pts = [t["position"][ids.index(str(row[c])), t["valid"][ids.index(str(row[c]))], :2]
               for c in ("first_id", "second_id") if str(row[c]) in ids]
        if not pts:
            near[n] = []
            continue
        tree = cKDTree(np.concatenate(pts))
        close = {m: int((tree.query(starts[m] * SNAP, distance_upper_bound=NEAR)[0] < NEAR).sum())
                 for m in members[place[k]] if m != n and len(starts[m])}
        near[n] = sorted((m for m, c in close.items() if c), key=lambda m: -close[m])
    pickle.dump(near, open(Path(args.pool) / "places.pkl", "wb"))
    sizes = np.array([len(v) for v in near.values()])
    print("%d scenes, %d places; other scenes pooled per scene: median %d, mean %.1f, none for %.0f%%" % (
        len(names), len(members), np.median(sizes), sizes.mean(), 100.0 * (sizes == 0).mean()))
    print("percentiles 25/75/90/99: %s" % np.percentile(sizes, [25, 75, 90, 99]).tolist())


def others_tracks(scene, others, folder, limit=MAX_OTHERS):
    """The vehicle tracks of up to ``limit`` of the ``others`` scenes, one dict per scene, without
    copies of tracks already taken (scenes cut from one drive overlap in time, and the judged pair
    must not come back as someone else)."""
    from scipy.spatial import cKDTree

    def sample(P, V):
        return [P[n, V[n], :2][::DUP_EVERY] for n in range(len(P))]

    seen = [x for x in sample(scene.position, scene.valid) if len(x)]
    out = []
    for m in others[:limit]:
        t = load_tracks(folder, m)
        tree = cKDTree(np.concatenate(seen)) if seen else None
        pts = sample(t["position"], t["valid"])
        keep = [n for n, x in enumerate(pts) if len(x) and (tree is None or
                (tree.query(x, distance_upper_bound=DUP_DIST)[0] < DUP_DIST).sum() < min(3, len(x)))]
        if keep:
            out.append({k: v[keep] for k, v in t.items()})
            seen += [pts[n] for n in keep]
    return out


def pooled(scene, groups):
    """``scene`` with the vehicles of ``groups`` (others_tracks) appended as extra agents of type
    VEHICLE over the same steps; the scene's own agents keep their indices."""
    if not groups:
        return scene
    T = scene.n_steps
    cat = lambda key: np.concatenate([g[key][:, :T] for g in groups])
    size = cat("size")
    out = copy.copy(scene)
    out.track_ids = list(scene.track_ids) + ["pool/%s" % i for g in groups for i in g["ids"]]
    out.types = list(scene.types) + ["VEHICLE"] * len(size)
    out.position = np.concatenate([scene.position, cat("position").astype(scene.position.dtype)])
    out.heading = np.concatenate([scene.heading, cat("heading").astype(scene.heading.dtype)])
    out.velocity = np.concatenate([scene.velocity, cat("velocity").astype(scene.velocity.dtype)])
    out.valid = np.concatenate([scene.valid, cat("valid")])
    out.length = np.concatenate([scene.length, size[..., 0]])
    out.width = np.concatenate([scene.width, size[..., 1]])
    out.height = np.concatenate([scene.height, size[..., 2]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--scenes", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--part")
    i = sub.add_parser("index")
    i.add_argument("--pool", required=True)
    i.add_argument("--pairs", required=True)
    args = ap.parse_args()
    {"extract": cmd_extract, "index": cmd_index}[args.cmd](args)


if __name__ == "__main__":
    main()
