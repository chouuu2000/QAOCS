import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
from scipy.stats import gaussian_kde
def calculate_si(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    si = np.std(np.sqrt(sobel_x**2 + sobel_y**2))
    return si

def calculate_ti(prev_frame, current_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = current_gray - prev_gray
    ti = np.std(diff)
    return ti

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    si_values = []
    ti_values = []
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap.get(cv2.CAP_PROP_FPS) 
  
    # calculate duration of the video 
    seconds = round(frames / fps) 
    video_time = datetime.timedelta(seconds=seconds) 
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        si = calculate_si(frame)
        si_values.append(si)

        if prev_frame is not None:
            ti = calculate_ti(prev_frame, frame)
            ti_values.append(ti)

        prev_frame = frame
    # get the video duration in seconds
    cap.release()
    return si_values, ti_values, video_time

def plot_si_ti(si_values, ti_values,video_name):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(si_values, label='Spatial Information (SI)')
    plt.xlabel('Frame Number')
    plt.ylabel('SI')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ti_values, label='Temporal Information (TI)', color='orange')
    plt.xlabel('Frame Number')
    plt.ylabel('TI')
    plt.legend()

    plt.suptitle('SI and TI Values over Frames')
    plt.savefig(os.path.join(save_path, video_name + "si_ti_plot.pdf"))
    plt.close()

    # plot Si and Ti values in the same plot X-axis SI, Y-axis TI
    xy = np.vstack([si_values, ti_values])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 6))
    plt.scatter(si_values, ti_values, c=z, s=10, edgecolor='none')
    plt.colorbar(label='Density')
    plt.xlabel('Spatial Information (SI)', fontsize=18)
    plt.ylabel('Temporal Information (TI)',fontsize=18)
    plt.title('SI and TI Values',fontsize=18)
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(save_path, f"{video_name}.pdf"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir_path", type=str, default="Video_source")
    args = parser.parse_args()
    store_figures_path = "./Figures"
    if not os.path.exists(store_figures_path):
        os.makedirs(store_figures_path)

    for video in os.listdir(args.video_dir_path):
        if video.endswith(".mp4"):
            video_path = os.path.join(args.video_dir_path, video)
            video_name = video.split(".")[0]
            save_path = os.path.join(store_figures_path, video_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            si_values, ti_values,duration = process_video(video_path)
            print(f"Video: {video_name} Duration: {duration} seconds Average SI: {np.mean(si_values)} TI: {np.mean(ti_values)}")
            plot_si_ti(si_values, ti_values,video_name)
