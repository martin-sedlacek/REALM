"""Crop the WRIST press clip onto the fingertips, magnify it, and put two runs side by side.

Why this exists
---------------
`ee_press_compliance.py` now records the wrist view as well as `external_sensor0`, and the wrist view
is the one that shows an inward TIP CURL: the camera is rigidly attached to the hand, so the fingers
are STATIC in that frame for the whole descent and the only thing that moves is the deflection
itself. (The standing "the wrist camera looks along the fingers and hides bending" advice is about
the SQUEEZE case, where the pads move within the plane the camera looks along.) But the motion is
0.1-0.4 mm on a 1280x720 frame, so it has to be cropped to the tips and magnified before anyone can
see it, and the numbers have to travel with the pixels.

    python scripts/debug_probes/curl_wrist_zoom.py \
        --runs curl_WRIST_kp15:DROID_robolab_padspring_kp15_ee_control:pad-spring\ kp15 \
               curl_WRIST_default:DROID_robolab_v2_ee_control:default \
        --out-prefix curl_WRIST

Each --runs entry is `<video stem>:<tipheel npy stem>:<label>` -- two stems because the probe names
the mp4 from REALM_VIDTAG and the npy from REALM_ROBOT. Writes, per run, a magnified tip clip and a
rest-vs-press still, and for two or more runs a frame-synced side-by-side clip and still.

Run it in the container (`av` and moviepy live there). Nothing here measures anything: the direction
claim comes from the probe's tip/heel numbers, which are burned into every frame so that the clip
cannot be quoted without them.
"""
import argparse
import os

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/logs/gripper_squeeze")
ap.add_argument("--runs", nargs="+", required=True, metavar="VIDSTEM:NPYSTEM:LABEL")
ap.add_argument("--video-suffix", default="_press_wrist.mp4")
ap.add_argument("--npy-suffix", default="_tipheel.npy")
ap.add_argument("--crop", default="620,300,1100,660",
                help="x0,y0,x1,y1 in the 1280x720 wrist frame. Default brackets the fingertips of "
                     "the robolab 2F-85 with the jaws shut; check one still before trusting it.")
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--out-prefix", default="curl_WRIST")
args = ap.parse_args()

X0, Y0, X1, Y1 = (int(v) for v in args.crop.split(","))


def font_at(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def load(vid_stem, npy_stem):
    vp = os.path.join(args.root, vid_stem + args.video_suffix)
    with av.open(vp) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    np_path = os.path.join(args.root, npy_stem + args.npy_suffix)
    th = np.load(np_path) if os.path.exists(np_path) else None
    if th is None:
        print(f"  [warn] no {np_path}: frames will carry no numbers")
    return fr, th


def crop_scale(im):
    c = im[Y0:Y1, X0:X1]
    w = int(c.shape[1] * args.scale)
    h = int(c.shape[0] * args.scale)
    return np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((w, h), Image.LANCZOS))


def band(im, lines, colour=(255, 235, 120)):
    img = Image.fromarray(np.ascontiguousarray(im))
    d = ImageDraw.Draw(img)
    f = font_at(max(15, im.shape[0] // 26))
    n = len(lines)
    d.rectangle([0, 0, img.size[0], 8 + n * (f.size + 4)], fill=(0, 0, 0))
    d.multiline_text((8, 4), "\n".join(lines), fill=colour, font=f)
    return np.asarray(img)


runs = []
for spec in args.runs:
    vid, npy, label = spec.split(":", 2)
    fr, th = load(vid, npy)
    print(f"{label}: {len(fr)} frames from {vid}{args.video_suffix}"
          + (f", {th.shape[1]} tip/heel samples" if th is not None else ""))
    runs.append(dict(vid=vid, label=label, fr=fr, th=th))

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # noqa: E402


def numbers(r, i):
    """The tip/heel deltas at frame i, relative to the run's own rest value (frame 0 of the hold)."""
    th = r["th"]
    if th is None or i >= th.shape[1]:
        return ["(no tip/heel trace)"]
    tip, heel = th[0], th[1]
    # The reference is the FIRST FINITE sample, not tip[0]: the probe records nan for every step
    # before capture_reference_geometry() has run (the jaw-settle phase), so tip[0] is nan and
    # subtracting it makes every frame read nan -- which is exactly what the first still did.
    fin = np.flatnonzero(np.isfinite(tip))
    if fin.size == 0:
        return ["(tip/heel trace is all nan)"]
    r0 = fin[0]
    dt = (tip[i] - tip[r0]) * 1000.0
    dh = (heel[i] - heel[r0]) * 1000.0
    if not np.isfinite(dt):
        return ["(before the reference pose)"]
    return [f"tip {dt:+.3f} mm    heel {dh:+.3f} mm",
            "tip DOWN + heel UP = tips curl INWARD"]


for r in runs:
    ims = [band(crop_scale(f), [f"{r['label']}   frame {i}"] + numbers(r, i))
           for i, f in enumerate(r["fr"])]
    out = os.path.join(args.root, f"{args.out_prefix}_{r['label'].replace(' ', '_')}_TIPZOOM.mp4")
    ImageSequenceClip(ims, fps=args.fps).write_videofile(out, codec="libx264", audio=False,
                                                         logger=None)
    print(f"wrote {out}  ({len(ims)} frames)")
    still = np.concatenate([ims[0], ims[-1]], axis=1)
    sp = os.path.join(args.root, f"{args.out_prefix}_{r['label'].replace(' ', '_')}_REST_vs_PRESS.png")
    Image.fromarray(still).save(sp)
    print(f"wrote {sp}  (rest | press)")

if len(runs) > 1:
    # Frame-synced: the phases are the same length in both runs because they come from the same probe
    # with the same env defaults, so index i is the same phase. Trim to the shorter one and say so.
    N = min(len(r["fr"]) for r in runs)
    if len({len(r["fr"]) for r in runs}) > 1:
        print(f"  note: frame counts differ {[len(r['fr']) for r in runs]}; trimming to {N}")
    seq = []
    for i in range(N):
        seq.append(np.concatenate(
            [band(crop_scale(r["fr"][i]), [f"{r['label']}  frame {i}"] + numbers(r, i))
             for r in runs], axis=1))
    out = os.path.join(args.root, f"{args.out_prefix}_SBS.mp4")
    ImageSequenceClip(seq, fps=args.fps).write_videofile(out, codec="libx264", audio=False,
                                                         logger=None)
    print(f"wrote {out}  ({N} frames, {' | '.join(r['label'] for r in runs)})")
    Image.fromarray(np.concatenate([seq[0], seq[-1]], axis=0)).save(
        os.path.join(args.root, f"{args.out_prefix}_SBS_REST_vs_PRESS.png"))
    print(f"wrote {args.root}/{args.out_prefix}_SBS_REST_vs_PRESS.png  (top rest, bottom press)")
print("WRIST_ZOOM_OK")
