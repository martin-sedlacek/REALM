#!/usr/bin/env python
"""Host-side (no container, no GPU): turn a droid100_first_frames run into something reviewable.

The probe leaves 100 three-view panels and 16 shard JSONs. Clicking through 100 images is not a
review, so this collapses them into contact sheets (10 tasks per sheet, stacked panels) plus a
REVIEW.md table sorted so that everything the probe flagged is at the top.

    python scripts/debug_probes/droid100_frames_sheet.py --run <REALM_LOGS>/droid100_first_frames/<RUN_ID>

Writes into the run dir: sheet_00.jpg .. sheet_NN.jpg, REVIEW.md, index.json (all records merged).
Reads only what the probe wrote; safe to re-run while shards are still filling.
"""
import argparse
import glob
import json
import os

from PIL import Image

#: Tasks per contact sheet. 10 panels of 1920x422 stack to 1920x4220, which a viewer can scroll.
PER_SHEET = 10
SHEET_WIDTH = 1600

#: Height of the DROID base column (realm/environments/constants.DROID_BASE_HEIGHT). The task
#: configs' robot position is the BOTTOM of the column, so reachability has to be measured from
#: robot_pos + this, not from robot_pos. Measuring from the floor overstates every distance by
#: enough to flag reachable tasks as unreachable (1.08 m vs the true 0.79 m on 003).
DROID_BASE_HEIGHT = 0.86244

#: Franka Panda's reach envelope. Beyond this the arm cannot touch the object at all, whatever the
#: policy does, so the task is unwinnable rather than hard.
REACH_LIMIT = 0.85

#: How far an object may sink between its authored position and where it settles before that counts
#: as a fall rather than a settle. OmniGibson raises every object by env.initial_pos_z_offset (0.2 m
#: in REALM's config) at load and it drops straight back, so ~0.15-0.20 m of "drop" is the NORMAL
#: outcome and flagging it buries the real falls. Measured on this run: 0.151 / 0.182 / 0.188 for
#: tasks that are demonstrably sitting on the table, against 0.83-0.89 for ones on the floor.
SETTLE_TOLERANCE = 0.30

#: An object whose AABB bottom is below this is on the floor, not on any tabletop in this suite
#: (every support region in scenes.yaml sits at z >= 0.5).
FLOOR_Z = 0.40


def arm_base(rec):
    """Where the arm actually starts: the task's robot position, raised by the column when mounted."""
    x, y, z = rec["robot_pos"]
    return (x, y, z + (DROID_BASE_HEIGHT if rec.get("use_droid_with_base") else 0.0))


def _dist(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5


def rescore(rec):
    """Recompute the review flags from the record, and attach the derived distances.

    The probe's own `flags` are a first pass written next to the pictures; these are the ones the
    review acts on. Both of the probe's numeric thresholds were wrong on contact with real data (see
    SETTLE_TOLERANCE and DROID_BASE_HEIGHT above), and every input is recorded, so they are fixed
    here -- host-side, re-runnable, no GPU -- rather than by re-rendering 100 tasks.
    """
    if rec["status"] != "ok":
        return rec
    base = arm_base(rec)
    objs = rec.get("objects", {})
    flags = []

    main = (objs.get("main") or [None])[0]
    if main:
        rec["reach_from_base"] = round(_dist(base, main["aabb_center"]), 4)
        rec["reach_from_ee"] = round(_dist(rec["ee_pos"], main["aabb_center"]), 4)
        if rec["reach_from_base"] > REACH_LIMIT:
            flags.append(f"UNREACHABLE:{rec['reach_from_base']:.2f}m")

    for role in ("main", "target", "distractors"):
        for o in objs.get(role, []):
            bottom = o["aabb_center"][2] - o["aabb_extent"][2] / 2
            drop = (o.get("drift") or [0, 0, 0])[2]
            if bottom < FLOOR_Z:
                flags.append(f"ON_FLOOR:{o['name']}")
            elif drop < -SETTLE_TOLERANCE:
                flags.append(f"FELL:{o['name']}({drop:+.2f}m)")

    for cam, stats in rec.get("view_stats", {}).items():
        if stats["mean"] < 4.0:
            flags.append(f"BLACK:{cam}")
        elif stats["mean"] > 245.0:
            flags.append(f"BLOWN:{cam}")

    # The authoring contract: a put/stack task must NOT start with its object already placed, and a
    # pick/rotate task whose instruction says "remove X from Y" must start with X on or in Y.
    rel = rec.get("relations", {})
    placed = (rel.get("inside") is True) or (rel.get("on_top") is True)
    if rec["task_type"] in ("put", "stack") and placed:
        flags.append("ALREADY_SOLVED")
    if rec["task_type"] in ("pick", "rotate") and objs.get("target") and not placed:
        flags.append("NO_INITIAL_RELATION")
    if rec.get("progression_fraction"):
        flags.append(f"PROGRESS_AT_T0:{rec['progression_fraction']}")
    if rec.get("env_collision"):
        flags.append("ROBOT_IN_COLLISION")

    rec["flags_raw"] = rec.get("flags", [])
    rec["flags"] = flags
    return rec


def load_records(run_dir):
    """Every shard's records, merged and sorted by task name. Missing shards are simply absent."""
    records, shards = [], []
    for path in sorted(glob.glob(os.path.join(run_dir, "shard*.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        shards.append({"shard": doc.get("shard"), "n_tasks": len(doc.get("tasks", [])),
                       "n_records": len(doc.get("records", [])),
                       "aborted_after": doc.get("aborted_after")})
        for rec in doc.get("records", []):
            rec["_shard"] = doc.get("shard")
            records.append(rescore(rec))
    records.sort(key=lambda r: r["task"])
    return records, shards


def build_sheets(run_dir, records):
    """Stack each group of PER_SHEET panels into one scrollable JPEG."""
    made = []
    with_panels = [r for r in records
                   if os.path.isfile(os.path.join(run_dir, "frames", r["task"], "panel.jpg"))]
    for s in range(0, len(with_panels), PER_SHEET):
        group = with_panels[s:s + PER_SHEET]
        tiles = []
        for rec in group:
            im = Image.open(os.path.join(run_dir, "frames", rec["task"], "panel.jpg"))
            h = round(im.height * SHEET_WIDTH / im.width)
            tiles.append(im.convert("RGB").resize((SHEET_WIDTH, h), Image.BILINEAR))
        canvas = Image.new("RGB", (SHEET_WIDTH, sum(t.height for t in tiles)), (18, 18, 20))
        y = 0
        for t in tiles:
            canvas.paste(t, (0, y))
            y += t.height
        out = os.path.join(run_dir, f"sheet_{s // PER_SHEET:02d}.jpg")
        canvas.save(out, quality=88)
        made.append((out, [r["task"] for r in group]))
    return made


def write_review(run_dir, records, shards, sheets):
    """The table. Flagged and failed tasks first -- that ordering IS the review queue."""
    def key(r):
        if r["status"] != "ok":
            return (0, r["task"])
        return (1 if r.get("flags") else 2, r["task"])

    lines = ["# DROID100 tabletop -- initial conditions at t=0", "",
             f"Run: `{run_dir}`", ""]
    ok = [r for r in records if r["status"] == "ok"]
    flagged = [r for r in ok if r.get("flags")]
    lines += [f"- records: **{len(records)}**  ok: **{len(ok)}**  "
              f"errored: **{len(records) - len(ok)}**  flagged: **{len(flagged)}**", ""]
    for sh in shards:
        if sh["aborted_after"]:
            lines.append(f"- shard {sh['shard']}: **ABORTED** after `{sh['aborted_after']}` "
                         f"({sh['n_records']}/{sh['n_tasks']})")
        elif sh["n_records"] < sh["n_tasks"]:
            lines.append(f"- shard {sh['shard']}: incomplete {sh['n_records']}/{sh['n_tasks']}")
    lines += ["", "## Contact sheets", ""]
    for path, tasks in sheets:
        lines.append(f"- `{os.path.basename(path)}` -- {tasks[0]} .. {tasks[-1]}")
    counts = {}
    for r in ok:
        for f in r["flags"]:
            counts[f.split(":")[0]] = counts.get(f.split(":")[0], 0) + 1
    if counts:
        lines += ["", "## Flags by kind", ""]
        for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{k}` -- {n}")

    lines += ["", "## Tasks", "",
              "reach = arm base (robot_pos + base column) to the main object's AABB centre; "
              f"Franka's envelope is ~{REACH_LIMIT} m.", "",
              "| task | type | instruction | reach | ee->obj | in/on | flags |",
              "|---|---|---|--:|--:|---|---|"]
    for r in sorted(records, key=key):
        if r["status"] != "ok":
            last = (r.get("traceback", "").strip().splitlines() or ["?"])[-1]
            lines.append(f"| `{r['task']}` | - | - | - | - | - | **ERROR** {last[:110]} |")
            continue
        rel = r.get("relations", {})
        lines.append(
            f"| `{r['task']}` | {r['task_type']} | {r['instruction']} | "
            f"{r.get('reach_from_base', '?')} | {r.get('reach_from_ee', '?')} | "
            f"{rel.get('inside')}/{rel.get('on_top')} | "
            f"{', '.join(r.get('flags', [])) or ''} |")
    path = os.path.join(run_dir, "REVIEW.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="the run dir the probe wrote (holds shard*.json)")
    args = p.parse_args()

    records, shards = load_records(args.run)
    assert records, f"no shard*.json records under {args.run}"
    sheets = build_sheets(args.run, records)
    with open(os.path.join(args.run, "index.json"), "w") as fh:
        json.dump({"shards": shards, "records": records}, fh, indent=2)
    review = write_review(args.run, records, shards, sheets)

    ok = [r for r in records if r["status"] == "ok"]
    print(f"{len(records)} records ({len(ok)} ok), {len(sheets)} sheets -> {review}")
    for r in records:
        if r["status"] != "ok":
            print(f"  ERROR {r['task']}")
        elif r.get("flags"):
            print(f"  flag  {r['task']}: {','.join(r['flags'])}")


if __name__ == "__main__":
    main()
