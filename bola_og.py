import numpy as np

class Bola:
    def __init__(self, env, bitrates, segment_duration, sizes, buffer_target=20.0):
        self.bitrates = bitrates  # list in kbps
        self.segment_duration = segment_duration  # seconds
        self.buffer_target = buffer_target  # seconds
        self.sizes = sizes
        self.throughput = env.cooked_bw[env.mahimahi_ptr] * 1e6 / 8.0 # 當前吞吐量（byte/s）
        self.Q_ref =20.0  # 參考緩衝區大小（單位：秒），影響切換行為

        # 取出上一段品質與 throughput（Bytes/sec）
        self.lastindex = env.data["QUALITY_INDEX"][-1] if env.data["QUALITY_INDEX"] else 0
        self.previousBW = env.data["CURRENT_THROUGHPUT"][-1] if env.data["CURRENT_THROUGHPUT"] else 1e6

        self.MINIMUM_BUFFER_S = 5
        self.MINIMUM_BUFFER_PER_LEVEL_S = 2
        self.Vp = None
        self.gp = None
        self.utilities = self._calculate_utilities(bitrates)
        self._calculate_parameters()

    def _calculate_utilities(self, bitrates):
        utilities = [np.log(b) for b in bitrates]
        u0 = utilities[0]
        return [u - u0 + 1 for u in utilities]

    def _calculate_parameters(self):
        u = self.utilities
        highest_index = np.argmax(u)
        if highest_index == 0:
            self.Vp = 1
            self.gp = 0
            return
        buffer_target = max(
            self.buffer_target,
            self.MINIMUM_BUFFER_S + self.MINIMUM_BUFFER_PER_LEVEL_S * len(self.bitrates)
        )
        self.gp = (u[highest_index] - 1) / (buffer_target / self.MINIMUM_BUFFER_S - 1)
        self.Vp = self.MINIMUM_BUFFER_S / self.gp

    def abr(self, buffer_sec):
        best_score = -float('inf')
        best_index = 0

        for i in range(len(self.bitrates)):
            bitrate_bps = self.bitrates[i] * 1000  # Convert kbps to bps
            score = (self.Vp * (self.utilities[i] - 1 + self.gp) - buffer_sec/1000) / bitrate_bps
            print('score', i, ':', score)
            if score > best_score:
                best_score = score
                best_index = i

        # BOLA-O: 若欲升級，檢查是否 sustainable
        previous_kbps = self.previousBW * 8 / 1000  # Bytes/sec -> kbps
        if best_index > self.lastindex and self.bitrates[best_index] > previous_kbps:
            best_index = self.lastindex
        
        if buffer_sec/1000 < self.Q_ref and best_index > 0:
            print('buffer', buffer_sec/1000)
            best_index -= 1  # 當緩衝區過低時降低質量


        return best_index