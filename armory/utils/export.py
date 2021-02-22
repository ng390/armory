import os
import logging
import numpy as np
import ffmpeg
import pickle
import time
from PIL import Image
from scipy.io import wavfile
from collections import defaultdict

from armory.data.datasets import ImageContext, VideoContext, AudioContext, So2SatContext


logger = logging.getLogger(__name__)


class LogitsExporter:
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir
        self.saved_samples = defaultdict(int)
        self._make_output_dir()

    def _make_output_dir(self):
        assert os.path.exists(self.base_output_dir) and os.path.isdir(
            self.base_output_dir
        ), f"Directory {self.base_output_dir} does not exist"
        assert os.access(
            self.base_output_dir, os.W_OK
        ), f"Directory {self.base_output_dir} is not writable"
        self.output_dir = os.path.join(self.base_output_dir, "saved_logits")
        if os.path.exists(self.output_dir):
            logger.warning(
                f"Sample output directory {self.output_dir} already exists. Creating new directory"
            )
            self.output_dir = os.path.join(
                self.base_output_dir, f"saved_logits_{time.time()}"
            )
        os.mkdir(self.output_dir)

    def export(self, logits, true_labels, example_type):
        for logit, y in zip(logits, true_labels):
            np.save(
                os.path.join(
                    self.output_dir,
                    f"{self.saved_samples[example_type]}_logits_{example_type}.npy",
                ),
                logit,
            )
            np.save(
                os.path.join(
                    self.output_dir,
                    f"{self.saved_samples[example_type]}_true_label_{example_type}.npy",
                ),
                y,
            )
            self.saved_samples[example_type] += 1


class SampleExporter:
    def __init__(self, base_output_dir, context, num_samples):
        self.base_output_dir = base_output_dir
        self.context = context
        self.num_samples = num_samples
        self.saved_samples = 0
        self.output_dir = None
        self.y_dict = {}

        if isinstance(self.context, VideoContext):
            self.export_fn = self._export_video
        elif isinstance(self.context, ImageContext):
            self.export_fn = self._export_images
        elif isinstance(self.context, AudioContext):
            self.export_fn = self._export_audio
        elif isinstance(self.context, So2SatContext):
            self.export_fn = self._export_so2sat
        else:
            raise TypeError(
                f"Expected VideoContext, ImageContext, AudioContext, or So2SatContext, got {type(self.context)}"
            )
        self._make_output_dir()

    def export(self, x, x_adv, y, y_adv):

        if self.saved_samples < self.num_samples:

            self.y_dict[self.saved_samples] = {"ground truth": y, "predicted": y_adv}
            self.export_fn(x, x_adv)

        if self.saved_samples >= self.num_samples:
            with open(os.path.join(self.output_dir, "predictions.pkl"), "wb") as f:
                pickle.dump(self.y_dict, f)

    def _make_output_dir(self):
        assert os.path.exists(self.base_output_dir) and os.path.isdir(
            self.base_output_dir
        ), f"Directory {self.base_output_dir} does not exist"
        assert os.access(
            self.base_output_dir, os.W_OK
        ), f"Directory {self.base_output_dir} is not writable"
        self.output_dir = os.path.join(self.base_output_dir, "saved_samples")
        if os.path.exists(self.output_dir):
            logger.warning(
                f"Sample output directory {self.output_dir} already exists. Creating new directory"
            )
            self.output_dir = os.path.join(
                self.base_output_dir, f"saved_samples_{time.time()}"
            )
        os.mkdir(self.output_dir)

    def _export_images(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial images are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i.min() < 0.0 or x_i.max() > 1.0:
                logger.warning(
                    "Benign image out of expected range. Clipping to [0, 1]."
                )
            if x_adv_i.min() < 0.0 or x_adv_i.max() > 1.0:
                logger.warning(
                    "Adversarial image out of expected range. Clipping to [0, 1]."
                )

            if x_i.shape[-1] == 1:
                mode = "L"
                x_i_mode = np.squeeze(x_i, axis=2)
                x_adv_i_mode = np.squeeze(x_adv_i, axis=2)
            elif x_i.shape[-1] == 3:
                mode = "RGB"
                x_i_mode = x_i
                x_adv_i_mode = x_adv_i
            else:
                raise ValueError(f"Expected 1 or 3 channels, found {x_i.shape[-1]}")

            benign_image = Image.fromarray(
                np.uint8(np.clip(x_i_mode, 0.0, 1.0) * 255.0), mode
            )
            adversarial_image = Image.fromarray(
                np.uint8(np.clip(x_adv_i_mode, 0.0, 1.0) * 255.0), mode
            )
            benign_image.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_benign.png")
            )
            adversarial_image.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.png")
            )

            self.saved_samples += 1

    def _export_so2sat(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial images are different shapes: {x_i.shape} vs. {x_adv_i.shape}"

            np.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_benign.npy"), x
            )
            np.save(
                os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.npy"),
                x_adv,
            )
            self.saved_samples += 1

    def _export_audio(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial audio are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i.min() < -1.0 or x_i.max() > 1.0:
                logger.warning(
                    "Benign audio out of expected range. Clipping to [-1, 1]"
                )
            if x_adv_i.min() < -1.0 or x_adv_i.max() > 1.0:
                logger.warning(
                    "Adversarial audio out of expected range. Clipping to [-1, 1]"
                )

            wavfile.write(
                os.path.join(self.output_dir, f"{self.saved_samples}_benign.wav"),
                rate=self.context.sample_rate,
                data=np.clip(x_i, -1.0, 1.0),
            )
            wavfile.write(
                os.path.join(self.output_dir, f"{self.saved_samples}_adversarial.wav"),
                rate=self.context.sample_rate,
                data=np.clip(x_adv_i, -1.0, 1.0),
            )

            self.saved_samples += 1

    def _export_video(self, x, x_adv):
        for x_i, x_adv_i in zip(x, x_adv):

            if self.saved_samples == self.num_samples:
                break

            assert np.all(
                x_i.shape == x_adv_i.shape
            ), f"Benign and adversarial videos are different shapes: {x_i.shape} vs. {x_adv_i.shape}"
            if x_i.min() < 0.0 or x_i.max() > 1.0:
                logger.warning("Benign video out of expected range. Clipping to [0, 1]")
            if x_adv_i.min() < 0.0 or x_adv_i.max() > 1.0:
                logger.warning(
                    "Adversarial video out of expected range. Clipping to [0, 1]"
                )

            folder = str(self.saved_samples)
            os.mkdir(os.path.join(self.output_dir, folder))

            benign_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{x_i.shape[2]}x{x_i.shape[1]}",
                )
                .output(
                    os.path.join(self.output_dir, folder, "video_benign.mp4"),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=self.context.frame_rate,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            adversarial_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{x_i.shape[2]}x{x_i.shape[1]}",
                )
                .output(
                    os.path.join(self.output_dir, folder, "video_adversarial.mp4"),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    r=self.context.frame_rate,
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )

            for n_frame, (x_frame, x_adv_frame) in enumerate(zip(x_i, x_adv_i)):

                benign_pixels = np.uint8(np.clip(x_frame, 0.0, 1.0) * 255.0)
                adversarial_pixels = np.uint8(np.clip(x_adv_frame, 0.0, 1.0) * 255.0)

                benign_image = Image.fromarray(benign_pixels, "RGB")
                adversarial_image = Image.fromarray(adversarial_pixels, "RGB")
                benign_image.save(
                    os.path.join(
                        self.output_dir, folder, f"frame_{n_frame:04d}_benign.png"
                    )
                )
                adversarial_image.save(
                    os.path.join(
                        self.output_dir, folder, f"frame_{n_frame:04d}_adversarial.png"
                    )
                )

                benign_process.stdin.write(benign_pixels.tobytes())
                adversarial_process.stdin.write(adversarial_pixels.tobytes())

            benign_process.stdin.close()
            benign_process.wait()
            adversarial_process.stdin.close()
            adversarial_process.wait()
            self.saved_samples += 1
