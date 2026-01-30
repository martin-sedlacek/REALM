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
        keys = results[0].keys()
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
        frame_img = np.concatenate((
            base_im,
            wrist_im,
        ), axis=1)

        if frame_img.dtype.kind == 'f':
             frame_img = (frame_img * 255).astype(np.uint8)
        elif frame_img.dtype != np.uint8:
             frame_img = frame_img.astype(np.uint8)

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