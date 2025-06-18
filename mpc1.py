import numpy as np
import itertools
import math
HORIZON = 5
PAST_BW_LEN = 8
M_IN_BYTES = 1000000
B_IN_MB = 1000000.0
MILLISECONDS_IN_SECOND = 1000.0
RESOLUTION_LIST = [(640, 360), (854, 480), (1280, 720), (1920, 1080)]


class MPC:
    def __init__(self, bitrates, B):
        self.past_bandwidths = []  # 用於存儲歷史頻寬
        self.bitrates = bitrates
        self.duration = B
    def estimate_bandwidth(self, simstate):
        """根據最近的下載數據估算可用頻寬"""
        if len(simstate.data["BYTES"]) == 0 or len(simstate.data["DELAY"]) == 0:
            return None  # 如果沒有數據，無法估算頻寬

        # 計算瞬時頻寬 (Mbps)
        last_video_chunk_size = simstate.data["BYTES"][-1] * 8  # 轉換為 bits
        last_video_chunk_delay = simstate.data["DELAY"][-1] / MILLISECONDS_IN_SECOND  # 轉換為秒
        instant_bandwidth = last_video_chunk_size / last_video_chunk_delay / B_IN_MB  # Mbps

        # 更新歷史頻寬數據
        self.past_bandwidths.append(instant_bandwidth)
        if len(self.past_bandwidths) > 5:  # 只保留最近 5 次的數據
            self.past_bandwidths.pop(0)

        # 計算移動平均頻寬 (MBps)
        #avg_bandwidth = np.mean(self.past_bandwidths)
         # 加權平均（最近資料權重高）
        weights = np.arange(1, len(self.past_bandwidths) + 1)  #  [1, 2, 3, 4, 5]
        weights = weights / weights.sum()  # 正規化
        weighted_avg_bandwidth = np.sum(np.array(self.past_bandwidths) * weights)

        return weighted_avg_bandwidth

    # def update_bandwidth_record(self, measured_bw):
    #     self.past_bandwidths.pop(0)
    #     self.past_bandwidths.append(measured_bw)

    #     if self.past_bandwidths[-2] > 0:
    #         prev_pred = self._harmonic_mean(self.past_bandwidths[:-1])
    #         error = abs(prev_pred - measured_bw) / measured_bw
    #     else:
    #         error = 0
    #     self.past_errors.pop(0)
    #     self.past_errors.append(error)

    # def predict_bandwidth(self):
    #     valid_bw = [bw for bw in self.past_bandwidths if bw > 0]
    #     if not valid_bw:
    #         return 1.0  # default value in Mbps

    #     harmonic_bw = self._harmonic_mean(valid_bw)
    #     max_error = max(self.past_errors) if any(e >= 0 for e in self.past_errors) else 0
    #     return harmonic_bw / (1 + max_error)

    def abr(self, env, bitrates, sizes):
        # sizes: list of chunk sizes for 4 quality levels, in bytes
        # bitrates: corresponding bitrate value for QoE estimation
        predict_bw = self.estimate_bandwidth(env)  # in Mbps
        if predict_bw  == None:
            return 0
        print("est_bw", predict_bw)

        all_combos = list(itertools.product(range(len(sizes)), repeat=HORIZON))

        best_qoe = -np.inf
        best_combo = None

        for combo in all_combos:
            temp_buffer = (env.buffer_size)/1000
            total_qoe = 0
            last_quality = combo[0]  # for smoothness penalty

            for i in range(HORIZON):
                quality = combo[i]
                chunk_size = sizes[quality]  # in bytes
                download_time = chunk_size*8 / (predict_bw * M_IN_BYTES)  # sec

                rebuffer = (max(download_time - temp_buffer, 0))
                temp_buffer = max(temp_buffer - download_time, 0) + self.duration/1000

                reward = bitrates[quality] - 5000 * rebuffer - abs(bitrates[quality] - bitrates[last_quality])
                #print("abr qoe:", reward)
                total_qoe += reward
                last_quality = quality

            if total_qoe > best_qoe:
                best_qoe = total_qoe
                best_combo = combo

        return best_combo[0]

    # def _harmonic_mean(self, values):
    #     return len(values) / sum(1.0 / v for v in values if v > 0)
