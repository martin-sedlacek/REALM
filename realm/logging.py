import numpy as np
import os
import csv
import shutil
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import omnigibson as og


def save_results_to_csv(results, log_dir, task, perturbation, filename=None):
    if filename is None:
        os.makedirs(log_dir, exist_ok=True)
        csv_results_filename = f"{log_dir}/{task}_{perturbation}.csv"
    else:
        csv_results_filename = filename
        os.makedirs(os.path.dirname(csv_results_filename), exist_ok=True)

    if len(results) > 0:
        keys = results[-1].keys()
        with open(csv_results_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    og.log.info(f"Saved run report to {csv_results_filename}")
    return csv_results_filename

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
            ImageSequenceClip(self.frame_filenames, fps=fps).write_videofile(save_filename + ".mp4", codec="libx264")
        else:
            if not self.frames:
                return
            ImageSequenceClip(self.frames, fps=fps).write_videofile(save_filename + ".mp4", codec="libx264")

    def cleanup(self):
        if self.disk_mode and os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)