import numpy as np

class Bola:
    def __init__(self, B):
        # BOLA 參數
        self.duration = B
        self.gamma = 5.0/B  # 權衡緩衝區大小和質量的參數 
        self.Q_ref =40.0  # 參考緩衝區大小（單位：秒），影響切換行為
        self.Vp = 1    # 稍後根據比特率和緩衝區動態調整
    
    def abr(self, env, bitrates_now, sizes_now):
        """
        參數:
        env: 從 core.py 傳入，提供必要的狀態資訊
        
        返回:
        quality_level: int, 0~3 表示選擇的質量等級
        """
        # 從環境中獲取當前狀態
        #bitrates = env.data["BITRATE"]  # 可用的比特率列表
        bitrates = bitrates_now
        buffer_size = env.buffer_size / 1000.0  # 將緩衝區大小轉換為秒
        #video_chunk_size = env.video_size[-1] if env.video_size else 0  # 當前分段大小（byte）
        video_chunk_size = sizes_now
        throughput = env.cooked_bw[env.mahimahi_ptr] * 1e6 / 8.0  # 當前吞吐量（byte/s）

        # 如果這是第一個分段，初始化參數
        if env.video_chunk_counter == 0:
            self.initialize_parameters(bitrates)

        # 確保 bitrates 不為空並且長度正確
        if not bitrates or len(bitrates) != 4:
            # 如果比特率數據不可用，選擇最低質量
            return 0

        # BOLA 的效用函數
        utilities = self.compute_bola_utility(bitrates, buffer_size, video_chunk_size)

        # 選擇效用最大的質量等級，但需考慮吞吐量限制
        quality_level = 0
        max_utility = -float('inf')

        for i in range(len(bitrates)):
            bitrate = bitrates[i]  # 單位：bps
            chunk_size = video_chunk_size  # 假設大小與前一個分段相似
            download_time = chunk_size[i] / throughput if throughput > 0 else float('inf')

            # 如果下載時間合理且效用更高，更新選擇
            #if utilities[i] > max_utility and download_time < buffer_size + env.service_time[-1] / 1000.0:
            if utilities[i] > max_utility and download_time < buffer_size + self.duration / 1000.0:
                max_utility = utilities[i]
                quality_level = i

        # 防止質量過高導致緩衝區耗盡
        if buffer_size < self.Q_ref / 2 and quality_level > 0:
            quality_level -= 1  # 當緩衝區過低時降低質量

        return quality_level

    def initialize_parameters(self, bitrates):
        """
        初始化 BOLA 參數，根據可用比特率設置 Vp
        """
        if not bitrates:
            self.Vp = 1.0  # 默認值
            return

        # Vp 是根據最大和最小比特率計算的參數
        r_min = min(bitrates) / 1e6  # 轉換為 Mbps
        r_max = max(bitrates) / 1e6  # 轉換為 Mbps
        self.Vp = (self.Q_ref - 1) / (np.log(r_max / r_min) + 5 )

    def compute_bola_utility(self, bitrates, buffer_size, video_chunk_size):
        """
        計算每個質量等級的 BOLA 效用值
        
        參數:
        bitrates: list, 可用的比特率 (bps)
        buffer_size: float, 當前緩衝區大小 (秒)

        返回:
        utilities: list, 每個質量等級的效用值
        """
        utilities = []
        #for bitrate in bitrates:
        for i in range(0, len(bitrates)):
            bitrate = bitrates[i]
            r = bitrate / 1e6  # 轉換為 Mbps
            # BOLA 效用函數: Vp * (log(r) + gamma) - buffer_size 相關項
            #utility = self.Vp * (np.log(r / min(bitrates) * 1e6) + self.gamma) - (buffer_size / self.Q_ref)
            utility = (self.Vp * (np.log(r / min(bitrates) * 1e6) + 5) - buffer_size )/ (video_chunk_size[i])
            utilities.append(utility)
        return utilities
