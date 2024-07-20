import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import ffmpeg
from ffmpeg_quality_metrics import FfmpegQualityMetrics, VmafOptions

def encode_video(input_file, crf_value, output_file):
    ffmpeg.input(input_file, ss=10, t=20).output(output_file, vcodec='libx264', crf=crf_value, preset='medium', an=None).run(overwrite_output=True)
    return output_file

def get_bitrate(video_file):
    probe = ffmpeg.probe(video_file)
    bitrate = int(probe['format']['bit_rate'])
    return bitrate / 1000  # Convert to Kbps

def cut_reference_video(input_file, reference_cut):
    ffmpeg.input(input_file, ss=10, t=20).output(reference_cut, vcodec='copy', an=None).run(overwrite_output=True)
    return reference_cut

def calculate_vmaf(reference_video_path, video_path):
    ffqm = FfmpegQualityMetrics(reference_video_path, video_path)
    metrics = ffqm.calculate(
        ["vmaf"],
        VmafOptions(
            model_path='./vmaf/model/vmaf_float_v0.6.1.json'
        ),
    )
    vmaf_sum = 0.0
    vmaf_len = 0
    for metric in metrics['vmaf']:
        vmaf_sum += metric['vmaf']
        vmaf_len += 1

    return vmaf_sum / vmaf_len

def main():
    #input_file = "Video_source/bigbuckbunny360p24.mp4"  # Replace with your input video file
    input_file = "Video_source/Tears_of_Steel_360p24.mp4"  # Replace with your input video file
    reference_cut = "reference_cut.mp4"
    cut_reference_video(input_file, reference_cut)

    crf_values = range(18, 52, 2)  # CRF values from 18 to 50, step 2
    bitrates = []
    vmaf_scores = []

    for crf in crf_values:
        output_file = f"output_crf_{crf}.mp4"
        encode_video(input_file, crf, output_file)
        bitrate = get_bitrate(output_file)
        vmaf_score = calculate_vmaf(reference_cut, output_file)
        print(f"CRF: {crf}, Bitrate: {bitrate:.0f} Kbps, VMAF: {vmaf_score:.2f}")
        bitrates.append(bitrate)
        vmaf_scores.append(vmaf_score)
        os.remove(output_file)  # Remove the encoded file to save space

    os.remove(reference_cut)  # Clean up the reference cut file

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("CRF Value", fontsize=18)
    ax1.set_ylabel("Bitrate (Kbps)", color="tab:blue", fontsize=18)
    ax1.plot(crf_values, bitrates, 'o-', color="tab:blue", label="Bitrate")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.grid(True)

    # Add value labels for Bitrate
    for i, txt in enumerate(bitrates):
        ax1.annotate(f"{txt:.0f}", (crf_values[i], bitrates[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Fit exponential curve for Bitrate
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    initial_guesses = (bitrates[0], 0.1, bitrates[-1])
    popt, _ = curve_fit(exp_func, crf_values, bitrates, p0=initial_guesses)
    x_fit = np.linspace(min(crf_values), max(crf_values), 100)
    y_fit = exp_func(x_fit, *popt)
    ax1.plot(x_fit, y_fit, 'r--', label='Fitted Curve')

    # Add VMAF scores to the plot
    ax2 = ax1.twinx()
    ax2.set_ylabel("VMAF Score", color="tab:red", fontsize=18)
    ax2.plot(crf_values, vmaf_scores, 's-', color="tab:red", label="VMAF Score")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.tick_params(axis="y", labelsize=14)

    # Add value labels for VMAF
    for i, txt in enumerate(vmaf_scores):
        ax2.annotate(f"{txt:.2f}", (crf_values[i], vmaf_scores[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make space for the title
    plt.title("CRF vs Bitrate and VMAF on TOS 360p", fontsize=20, fontweight='bold')
    fig.legend(loc="upper center", bbox_to_anchor=(0.7, 0.9), ncol=2)
    plt.savefig("crf_vs_bitrate_vmaf.png")
    plt.savefig("crf_vs_bitrate_vmaf.pdf")
    plt.show()

if __name__ == "__main__":
    main()


