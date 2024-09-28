"""
This file is included in this repository as an example of how to handle the data in the h5 files,
as they contain not only the video frames but the corner positions of each frame as well. The structure of this code was written by 
Yilun Wu. 
"""

# The Kaggle environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# In Kaggle, input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/../Datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import cv2
from pathlib import Path
from tqdm import tqdm

#This function helps understand what the 
def h5_to_video(h5, output_dir, fps):
    output_dir.mkdir(parents=True, exist_ok=True)

    # get images and targets
    h5f = h5py.File(h5, "r")
    images = h5f["images"]
    targets = [h5f[f"targets/{i:05d}"][()] for i in range(len(images))]

    # draw targets on images
    frames = []
    for image, target in tqdm(zip(images, targets), total=len(images)):
        image = image.transpose(1, 2, 0)
        frame = image.copy()
        for gate in target:
            xy = gate.reshape(-1, 3)[..., :2] * image.shape[1::-1]
            visibility = gate.reshape(-1, 3)[..., 2]
            if np.all(visibility > 0):
                cv2.polylines(frame, [xy.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                # Draw lines between visible corners
                for i in range(len(xy) - 1):
                    if visibility[i] > 0 and visibility[i + 1] > 0:
                        cv2.line(frame, tuple(xy[i].astype(int)), tuple(xy[i + 1].astype(int)), color=(0, 255, 0), thickness=2)
                # Draw line from the last corner to the first to form a loop if both are visible
                if visibility[-1] > 0 and visibility[0] > 0:
                    cv2.line(frame, tuple(xy[-1].astype(int)), tuple(xy[0].astype(int)), color=(0, 255, 0), thickness=2)
        frames.append(frame)

    h5f.close()

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(output_dir / f"{h5.stem}.mp4"), codec="libx264")
h5_to_video(Path("../Datasets/autonomous_flight-01a-ellipse.h5"), Path('.'), 30)