"""Run artifacts: CSV reports, consolidated parquet trajectories/videos, and the video recorder.

Layout contract (frozen -- downstream tooling and tests/test_vector_integrity.py read it):
    reports/{task}_{perturbation}.csv          one row per rollout, written by save_results
    {qpos,actions}/{task}.parquet              one row per rollout, appended by append_trajectory
    videos/{task}.parquet                      one row per rollout, appended by append_video

The parquet "append" is read-concat-rewrite, so cost grows with rows already written -- fine at 25
repeats, worth a real row-group append if repeats ever grow by an order of magnitude.
"""
import numpy as np
import os
import csv
import shutil
import pandas as pd
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import omnigibson as og

#: Recorded videos are downscaled to this height (keeping aspect) before encoding.
VIDEO_TARGET_HEIGHT = 480


def save_results(results, log_dir, task, perturbation, filename=None):
    """Write the run report CSV: one row per rollout, minus the large array/bytes columns.

    Rewrites the file in full each call (the caller invokes this after every rollout, so a run that
    dies part way leaves a readable prefix). Column order comes from the entry dicts -- see
    realm/rollout.py::build_result_entry. Returns the CSV path, which the caller passes back in as
    @filename to keep appending to the same report.
    """
    if filename is None:
        os.makedirs(log_dir, exist_ok=True)
        base_filename = f"{log_dir}/{task}_{perturbation}"
    else:
        base_filename = os.path.splitext(filename)[0]
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)

    csv_filename = f"{base_filename}.csv"
    if len(results) > 0:
        # The trajectories and video bytes go to the parquets, not the report.
        csv_results = []
        for r in results:
            csv_row = {k: v for k, v in r.items() if k not in ["qpos", "actions", "video"]}
            csv_results.append(csv_row)

        if csv_results:
            keys = csv_results[-1].keys()
            with open(csv_filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(csv_results)
    og.log.info(f"Saved run report to {csv_filename}")
    return csv_filename


def _append_parquet_row(parquet_path, row):
    """Append one row (a plain dict) to @parquet_path, creating it if absent.

    Read-concat-rewrite, because parquet has no in-place append. A corrupted existing file is
    replaced rather than crashing a long eval on its final write.
    """
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    new_row = pd.DataFrame([row])

    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        except Exception as e:
            og.log.warning(f"Corrupted parquet at {parquet_path}, starting fresh: {e}")
            combined = new_row
    else:
        combined = new_row

    combined.to_parquet(parquet_path, index=False)


def append_trajectory(log_dir, task, perturbation, repeat, qpos_arr, actions_arr):
    """Append one repeat's qpos and actions to {qpos,actions}/{task}.parquet.

    Columns: task, perturbation, repeat, data (the trajectory as a nested list).
    """
    for subdir, arr in [("qpos", qpos_arr), ("actions", actions_arr)]:
        _append_parquet_row(
            os.path.join(log_dir, subdir, f"{task}.parquet"),
            {"task": task, "perturbation": perturbation, "repeat": repeat, "data": arr.tolist()},
        )


def append_video(log_dir, task, perturbation, repeat, video_bytes):
    """Append one repeat's encoded video bytes to videos/{task}.parquet. None is a silent no-op."""
    if video_bytes is None:
        return
    _append_parquet_row(
        os.path.join(log_dir, "videos", f"{task}.parquet"),
        {"task": task, "perturbation": perturbation, "repeat": repeat, "video": video_bytes},
    )


def _to_uint8(img):
    """@img as uint8: floats are assumed [0, 1] and scaled by 255, everything else is cast."""
    if img.dtype.kind == 'f':
        return (img * 255).astype(np.uint8)
    if img.dtype != np.uint8:
        return img.astype(np.uint8)
    return img


class VideoRecorder:
    """Accumulates one rollout's camera frames and encodes them to H.264 on demand.

    Frames are tiled per step: base | wrist side by side, or a 2x2 grid (base | second exterior /
    wrist | black) when a second exterior view is recorded. `disk_mode` spools frames to PNGs under
    @log_dir instead of holding them in memory -- call cleanup() afterwards to remove them.
    """

    def __init__(self, log_dir, timestamp, run_id, task=None, perturbation=None, disk_mode=False):
        self.disk_mode = disk_mode
        self.count = 0

        if self.disk_mode:
            suffix = ""
            if task:
                suffix += f"_{task}"
            if perturbation:
                suffix += f"_{perturbation}"
            self.temp_frame_dir = os.path.join(log_dir, f"{timestamp}_frames_{run_id}{suffix}")
            os.makedirs(self.temp_frame_dir, exist_ok=True)
            self.frame_filenames = []
        else:
            self.frames = []

    def _build_frame(self, base_im, wrist_im, base_im_second=None):
        """Tile one step's views into a single even-dimensioned uint8 frame of <= 480p height."""
        base_im = _to_uint8(base_im)
        wrist_im = _to_uint8(wrist_im)
        if base_im_second is not None:
            base_im_second = _to_uint8(base_im_second)

        # All tiles are brought to the base image's size before concatenation.
        target_size = (base_im.shape[1], base_im.shape[0])  # (width, height)

        if wrist_im.shape[:2] != base_im.shape[:2]:
            wrist_im = np.array(Image.fromarray(wrist_im).resize(target_size))

        if base_im_second is not None and base_im_second.shape[:2] != base_im.shape[:2]:
            base_im_second = np.array(Image.fromarray(base_im_second).resize(target_size))

        if base_im_second is not None:
            padding = np.zeros_like(base_im)
            top_row = np.concatenate((base_im, base_im_second), axis=1)
            bottom_row = np.concatenate((wrist_im, padding), axis=1)
            frame_img = np.concatenate((top_row, bottom_row), axis=0)
        else:
            frame_img = np.concatenate((base_im, wrist_im), axis=1)

        h, w = frame_img.shape[:2]
        if h > VIDEO_TARGET_HEIGHT:
            new_w = int(w * (VIDEO_TARGET_HEIGHT / h))
            frame_img = np.array(Image.fromarray(frame_img).resize((new_w, VIDEO_TARGET_HEIGHT)))

        # H.264 requires even dimensions.
        h, w = frame_img.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            new_h = h if h % 2 == 0 else h - 1
            new_w = w if w % 2 == 0 else w - 1
            frame_img = np.array(Image.fromarray(frame_img).resize((new_w, new_h)))

        return frame_img

    def add_frame(self, base_im, wrist_im, base_im_second=None):
        frame_img = self._build_frame(base_im, wrist_im, base_im_second)

        if self.disk_mode:
            frame_path = os.path.join(self.temp_frame_dir, f"frame_{self.count:05d}.png")
            Image.fromarray(frame_img).save(frame_path)
            self.frame_filenames.append(frame_path)
        else:
            self.frames.append(frame_img)

        self.count += 1

    def _build_clip(self, fps):
        """An ImageSequenceClip over the recorded frames, or None when nothing was recorded."""
        source = self.frame_filenames if self.disk_mode else self.frames
        if not source:
            return None
        return ImageSequenceClip(source, fps=fps)

    def save_video(self, save_filename, fps=15):
        save_dir = os.path.dirname(save_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        clip = self._build_clip(fps)
        if clip is None:
            return
        clip.write_videofile(save_filename + ".mp4", codec="libx264")

    def get_video_bytes(self, fps=15):
        """The recording encoded to mp4, as bytes -- what append_video stores. None if no frames."""
        clip = self._build_clip(fps)
        if clip is None:
            return None

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_name = tmp.name

        try:
            clip.write_videofile(tmp_name, codec="libx264", logger=None)
            with open(tmp_name, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def cleanup(self):
        """Remove the spooled PNG frames (disk_mode only)."""
        if self.disk_mode and os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)
