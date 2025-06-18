import numpy as np
import ppo2 as network

class pensieve:
    def __init__(self, env, B, video_bit_rates, model_path="pensieve_pretrain/nn_model_ep_341400.pth"):
        # 自動補成固定長度 6
        self.A_DIM = 6
        pad_len = self.A_DIM - len(video_bit_rates)
        self.video_bit_rates = video_bit_rates + [0] * pad_len
        self.valid_bitrate_mask = [1] * len(video_bit_rates) + [0] * pad_len

        self.S_INFO = 6
        self.S_LEN = 8
        self.BUFFER_NORM_FACTOR = 10.0
        self.CHUNK_TIL_VIDEO_END_CAP = 48.0
        self.M_IN_K = 1000.0

        self.model = network.Network(state_dim=[self.S_INFO, self.S_LEN],
                                     action_dim=self.A_DIM,
                                     learning_rate=0.0001)
        self.model.load_model(model_path)

        self.total_time = env.total_video_time
        if len(env.data["TIME"]) == 0:
            self.time_now = 0
        else:
            self.time_now = env.data["TIME"][-1]
        self.segment_length_now = B

        self.reset()

    def reset(self):
        self.state = np.zeros((self.S_INFO, self.S_LEN))
        self.last_bit_rate = 1
        self.time_stamp = 0

    def select_action(self, buffer_size, delay, video_chunk_size, next_video_chunk_sizes):
        # 補齊 chunk_sizes
        padded_chunk_sizes = next_video_chunk_sizes + [0] * (self.A_DIM - len(next_video_chunk_sizes))

        # 更新 state
        self.state = np.roll(self.state, -1, axis=1)
        self.state[0, -1] = self.video_bit_rates[self.last_bit_rate] / float(np.max(self.video_bit_rates))
        self.state[1, -1] = buffer_size / self.BUFFER_NORM_FACTOR
        if delay == 0 :
            self.state[2, -1] = 0
        else :
            self.state[2, -1] = float(video_chunk_size) / float(delay) / self.M_IN_K
        self.state[3, -1] = float(delay) / self.M_IN_K / self.BUFFER_NORM_FACTOR
        self.state[4, :self.A_DIM] = np.array(padded_chunk_sizes) / self.M_IN_K / self.M_IN_K

        video_chunk_remain_est = max((self.total_time - self.time_now) / self.segment_length_now, 1)
        self.state[5, -1] = min(video_chunk_remain_est, self.CHUNK_TIL_VIDEO_END_CAP) / self.CHUNK_TIL_VIDEO_END_CAP

        # 動作選擇 + mask 遮蔽非法 bitrate
        action_prob = self.model.predict(np.reshape(self.state, (1, self.S_INFO, self.S_LEN)))
        mask = np.array(self.valid_bitrate_mask)
        masked_logits = np.where(mask, np.log(action_prob), -np.inf)
        bit_rate = int(np.argmax(masked_logits + np.random.gumbel(size=len(action_prob))))

        self.last_bit_rate = bit_rate
        return bit_rate