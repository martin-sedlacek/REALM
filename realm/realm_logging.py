import numpy as np
import os
import csv
import shutil
import pandas as pd
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import omnigibson as og


def save_results(results, log_dir, task, perturbation, filename=None):
    if filename is None:
        os.makedirs(log_dir, exist_ok=True)
        base_filename = f"{log_dir}/{task}_{perturbation}"
    else:
        base_filename = os.path.splitext(filename)[0]
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)

    csv_filename = f"{base_filename}.csv"
    if len(results) > 0:
        # Filter out large data from CSV
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


def append_trajectory(log_dir, task, perturbation, repeat, qpos_arr, actions_arr):
    """Append one repeat's qpos and actions to consolidated parquets in log_dir.

    Each parquet is named after the task (e.g., task_name.parquet) and has 
    columns: task, perturbation, repeat, data (as nested list).
    """
    for subdir, arr in [("qpos", qpos_arr), ("actions", actions_arr)]:
        parquet_path = os.path.join(log_dir, subdir, f"{task}.parquet")
        os.makedirs(os.path.join(log_dir, subdir), exist_ok=True)

        new_row = pd.DataFrame([{
            "task": task,
            "perturbation": perturbation,
            "repeat": repeat,
            "data": arr.tolist(),
        }])

        if os.path.exists(parquet_path):
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row

        combined.to_parquet(parquet_path, index=False)


def append_video(log_dir, task, perturbation, repeat, video_bytes):
    """Append one repeat's video bytes to a consolidated parquet in log_dir/videos.
    Each parquet is named after the task (e.g., task_name.parquet).
    """
    if video_bytes is None:
        return

    parquet_path = os.path.join(log_dir, "videos", f"{task}.parquet")
    os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)

    new_row = pd.DataFrame([{
        "task": task,
        "perturbation": perturbation,
        "repeat": repeat,
        "video": video_bytes,
    }])

    if os.path.exists(parquet_path):
        existing = pd.read_parquet(parquet_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_parquet(parquet_path, index=False)


class VideoRecorder:
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
        # Ensure images are uint8
        if base_im.dtype.kind == 'f':
            base_im = (base_im * 255).astype(np.uint8)
        elif base_im.dtype != np.uint8:
            base_im = base_im.astype(np.uint8)

        if wrist_im.dtype.kind == 'f':
            wrist_im = (wrist_im * 255).astype(np.uint8)
        elif wrist_im.dtype != np.uint8:
            wrist_im = wrist_im.astype(np.uint8)

        if base_im_second is not None:
            if base_im_second.dtype.kind == 'f':
                base_im_second = (base_im_second * 255).astype(np.uint8)
            elif base_im_second.dtype != np.uint8:
                base_im_second = base_im_second.astype(np.uint8)

        # Check if resizing is needed
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

        # Downsize to 480p
        target_height = 480
        h, w = frame_img.shape[:2]
        if h > target_height:
            new_w = int(w * (target_height / h))
            frame_img = np.array(Image.fromarray(frame_img).resize((new_w, target_height)))

        # Ensure dimensions are even for H.264 compatibility
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

    def save_video(self, save_filename, fps=15):
        save_dir = os.path.dirname(save_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if self.disk_mode:
            if not self.frame_filenames:
                return
            clip = ImageSequenceClip(self.frame_filenames, fps=fps)
        else:
            if not self.frames:
                return
            clip = ImageSequenceClip(self.frames, fps=fps)
        
        clip.write_videofile(save_filename + ".mp4", codec="libx264")

    def get_video_bytes(self, fps=15):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_name = tmp.name
        
        try:
            if self.disk_mode:
                if not self.frame_filenames:
                    return None
                clip = ImageSequenceClip(self.frame_filenames, fps=fps)
            else:
                if not self.frames:
                    return None
                clip = ImageSequenceClip(self.frames, fps=fps)
            
            clip.write_videofile(tmp_name, codec="libx264", logger=None)
            with open(tmp_name, "rb") as f:
                video_bytes = f.read()
            return video_bytes
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def cleanup(self):
        if self.disk_mode and os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)
