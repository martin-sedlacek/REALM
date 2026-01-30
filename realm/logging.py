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
    def __init__(self, log_dir, timestamp, run_id):
        self.temp_frame_dir = os.path.join(log_dir, f"{timestamp}_frames_{run_id}")
        os.makedirs(self.temp_frame_dir, exist_ok=True)
        self.frame_filenames = []
        self.count = 0

    def add_frame(self, base_im, wrist_im):
        # Ensure images are uint8
        if base_im.dtype.kind == 'f':
            base_im = (base_im * 255).astype(np.uint8)
        elif base_im.dtype != np.uint8:
            base_im = base_im.astype(np.uint8)

        if wrist_im.dtype.kind == 'f':
            wrist_im = (wrist_im * 255).astype(np.uint8)
        elif wrist_im.dtype != np.uint8:
            wrist_im = wrist_im.astype(np.uint8)

        # Check if resizing is needed
        if base_im.shape[:2] != wrist_im.shape[:2]:
            base_pixels = base_im.shape[0] * base_im.shape[1]
            wrist_pixels = wrist_im.shape[0] * wrist_im.shape[1]

            if base_pixels > wrist_pixels:
                # Resize base to match wrist
                new_size = (wrist_im.shape[1], wrist_im.shape[0])
                base_im = np.array(Image.fromarray(base_im).resize(new_size))
            else:
                # Resize wrist to match base
                new_size = (base_im.shape[1], base_im.shape[0])
                wrist_im = np.array(Image.fromarray(wrist_im).resize(new_size))

        frame_img = np.concatenate((
            base_im,
            wrist_im,
        ), axis=1)

        # Ensure dimensions are even for H.264 compatibility
        h, w = frame_img.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            new_h = h if h % 2 == 0 else h - 1
            new_w = w if w % 2 == 0 else w - 1
            frame_img = np.array(Image.fromarray(frame_img).resize((new_w, new_h)))

        frame_path = os.path.join(self.temp_frame_dir, f"frame_{self.count:05d}.png")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        Image.fromarray(frame_img).save(frame_path)
        self.frame_filenames.append(frame_path)
        self.count += 1

    def save_video(self, save_filename, fps=15):
        if not self.frame_filenames:
            return
        save_dir = os.path.dirname(save_filename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        ImageSequenceClip(self.frame_filenames, fps=fps).write_videofile(save_filename + ".mp4", codec="libx264")

    def cleanup(self):
        if os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)