import numpy as np
import logging
B_IN_MB = 1000000.0
MILLISECONDS_IN_SECOND = 1000.0
RESOLUTION_LIST = [(640, 360), (854, 480), (1280, 720), (1920, 1080)]

class RateBasedABR:
    def __init__(self):
        self.logger = logging.getLogger("RateBasedABR")
        self.logger.setLevel(logging.ERROR)
        self.past_bandwidths = []  # 用於存儲歷史頻寬

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
        avg_bandwidth = np.mean(self.past_bandwidths)
        return avg_bandwidth

    def select_quality(self, simstate):
        """根據估算的頻寬選擇適當的影片品質 (0,1,2,3)"""
        estimated_bandwidth = self.estimate_bandwidth(simstate)
        if estimated_bandwidth is None:
            return 0  # 如果無法估算頻寬，選擇最低品質

        # 將 Mbps 轉換為 kbps (1 Mbps = 1000 kbps)
        estimated_bandwidth_kbps = estimated_bandwidth * 1000

        # 獲取當前的碼率對應表
        bitrates = simstate.data["BITRATE"]
        # if len(bitrates) < 4:
        #     return 0  # 確保碼率數據完整

        # 選擇最接近但不超過當前頻寬的碼率
        selected_quality = 0
        for i in range(len(bitrates)):
            if estimated_bandwidth_kbps >= bitrates[i]:
                selected_quality = i
            else:
                break

        # 確保 selected_quality 不超過 bitrates 或 RESOLUTION_LIST
        #selected_quality = min(selected_quality, len(bitrates) - 1, len(RESOLUTION_LIST) - 1)
        selected_quality = min(selected_quality, 3)

        return selected_quality  # 返回對應的品質等級 (0,1,2,3)

    def abr(self, simstate):
        """對應 ABR 的主要調用方法，返回下一個 chunk 的品質等級"""
        return self.select_quality(simstate)
