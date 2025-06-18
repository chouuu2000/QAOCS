import numpy as np

MPC_PREDICT_LEN = 5  # 預測幾個未來 chunk
BITS_IN_BYTE = 8.0
MILLISECONDS_IN_SECOND = 1000.0


class MPC:
    def __init__(self, chunk_duration=4000.0, alpha=0.5, beta1=0.1, beta2=-1.0, gamma=-30.0):
        self.chunk_duration = chunk_duration  # 每段影片片段長度（毫秒）
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma

    def abr(self, env):
        assert hasattr(env, "buffer_size") and hasattr(env, "cooked_bw")
        current_buffer = env.buffer_size
        past_vmaf = env.vmaf[-1] if len(env.vmaf) > 0 else 70

        # 模擬所有畫質組合策略（每個 chunk 有 N 個畫質）
        num_levels = len(env.RESOLUTION_LIST)
        all_combos = self.enumerate_combinations(num_levels, MPC_PREDICT_LEN)

        best_qoe = float('-inf')
        best_combo = [0] * MPC_PREDICT_LEN

        for combo in all_combos:
            buffer = current_buffer
            vmaf = past_vmaf
            qoe_sum = 0

            for i in range(MPC_PREDICT_LEN):
                level = combo[i]
                bitrate = env.last_bitrates[level]  # 預先記錄最後一個 chunk 各畫質對應的 bitrate
                size = bitrate * self.chunk_duration / MILLISECONDS_IN_SECOND / BITS_IN_BYTE  # byte

                # 根據目前估算的 bandwidth 計算下載 delay（簡化）
                est_bandwidth = np.mean(env.cooked_bw[max(0, env.mahimahi_ptr - 5):env.mahimahi_ptr]) * 1e6 / BITS_IN_BYTE  # Bytes/s
                delay = size / est_bandwidth * MILLISECONDS_IN_SECOND
                stall = max(0, delay - buffer)
                buffer = max(0, buffer - delay) + self.chunk_duration

                # VMAF 模擬（假設高畫質 VMAF 越高）
                vmaf_next = 70 + 10 * level
                quality_score = self.alpha * (vmaf_next - 70)
                smoothness_score = self.beta1 * (vmaf_next - vmaf) if vmaf_next >= vmaf else self.beta2 * (vmaf_next - vmaf)
                stall_score = self.gamma * (stall / 1000.0)

                qoe = quality_score + smoothness_score + stall_score
                qoe_sum += qoe
                vmaf = vmaf_next

            if qoe_sum > best_qoe:
                best_qoe = qoe_sum
                best_combo = combo

        # 回傳當前應該選的畫質等級
        return best_combo[0]

    def enumerate_combinations(self, num_levels, horizon):
        return np.array(np.meshgrid(*[range(num_levels)] * horizon)).T.reshape(-1, horizon)
