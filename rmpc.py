import numpy as np
import itertools

HORIZON = 5
PAST_BW_LEN = 8
M_IN_BYTES = 1000000.0
B_IN_MB = 1000000.0
MILLISECONDS_IN_SECOND = 1000.0


class MPC:
    def __init__(self):
        self.past_bandwidths = []
        self.past_bandwidth_ests = []
        self.past_errors = []

    def estimate_bandwidth(self, simstate):
        if len(simstate.data["BYTES"]) == 0 or len(simstate.data["DELAY"]) == 0:
            return None

        # 當前 chunk 實測帶寬 (Mbps)
        size = simstate.data["BYTES"][-1] * 8  # bits
        delay = simstate.data["DELAY"][-1] / MILLISECONDS_IN_SECOND  # sec
        raw_bw = size / delay / B_IN_MB  # Mbps

        # 計算與前一次預測的誤差
        if self.past_bandwidth_ests:
            prev_est = self.past_bandwidth_ests[-1]
            err = abs(prev_est - raw_bw) / raw_bw
        else:
            err = 0

        # 更新紀錄
        self.past_bandwidths.append(raw_bw)
        self.past_bandwidth_ests.append(raw_bw)
        self.past_errors.append(err)

        # 截斷長度
        self.past_bandwidths = self.past_bandwidths[-PAST_BW_LEN:]
        self.past_bandwidth_ests = self.past_bandwidth_ests[-PAST_BW_LEN:]
        self.past_errors = self.past_errors[-PAST_BW_LEN:]

        # 調和平均
        valid_bw = [b for b in self.past_bandwidths if b > 0]
        if not valid_bw:
            return 1.0
        harmonic_bw = 1.0 / np.mean([1.0 / b for b in valid_bw])

        # 根據最大誤差做保守修正
        max_error = max(self.past_errors)
        robust_bw = harmonic_bw / (1 + max_error)
        return robust_bw

    def abr(self, env, bitrates, sizes, B):
        predict_bw = self.estimate_bandwidth(env)
        if predict_bw is None:
            return 0

        print(f"[MPC] Robust bandwidth estimate: {predict_bw:.2f} Mbps")
        all_combos = list(itertools.product(range(len(sizes)), repeat=HORIZON))

        best_qoe = -np.inf
        best_combo = None

        for combo in all_combos:
            temp_buffer = env.buffer_size / 1000.0  # ms -> sec
            total_qoe = 0
            last_quality = combo[0]

            for i in range(HORIZON):
                quality = combo[i]
                chunk_size = sizes[quality]  # bytes
                download_time = chunk_size * 8 / (predict_bw * M_IN_BYTES)  # sec
                rebuffer = max(download_time - temp_buffer, 0)
                temp_buffer = max(temp_buffer - download_time, 0) + B / 1000.0

                # QoE 計算：normalize + penalty
                bitrate_score = bitrates[quality] / 1000.0
                rebuffer_penalty = 4.3 * rebuffer
                smoothness_penalty = abs(bitrates[quality] - bitrates[last_quality]) / 1000.0
                reward = bitrate_score - rebuffer_penalty - smoothness_penalty

                total_qoe += reward
                last_quality = quality

            if total_qoe > best_qoe:
                best_qoe = total_qoe
                best_combo = combo

        return best_combo[0]

    def reset(self):
        self.past_bandwidths = []
        self.past_bandwidth_ests = []
        self.past_errors = []
