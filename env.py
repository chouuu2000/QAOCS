import numpy as np
import core as abrenv
import load_trace
from video import Video
from core import S_INFO, S_LEN,A_DIM 
# download time in seconds,buffer size in 10 sec, throughput in kbps, chunk duration in ms, chunk CRF
# video features: SI, TI, GLCM, network condition histogram
#S_INFO = 9
#S_LEN = 8  # take how many frames in the past
#A_DIM = 2
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100

BUFFER_NORM_FACTOR = 10.0
SI_TI_NORM_FACTOR = 100.0
AVG_CONTRAST_NORM_FACTOR = 50.0
AVG_DISSIMILARITY_NORM_FACTOR = 1
AVG_HOMOGENEITY_NORM_FACTOR = 0.5
AVG_ENERGY_NORM_FACTOR = 0.1
AVG_CORRELATION_NORM_FACTOR = 1
AVG_ASM_NORM_FACTOR = 0.01

M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6


class ABREnv():
    def __init__(self, random_seed=RANDOM_SEED, trace_folder=None,video_path=None,writer=None, test=False):

        #video_path = 'bigbuckbunny360p24.mp4' if video_path is None else video_path
        #video_path = 'bigbuckbunny1080p30.mp4' 
        video_path = 'Video_source/LOL_3D.mp4' if video_path is None else video_path
        np.random.seed(random_seed)
        if test:
            all_cooked_time, all_cooked_bw,all_trace_file_names = load_trace.load_trace(trace_folder)
        else:
            all_cooked_time, all_cooked_bw, all_trace_file_names = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          all_trace_file_names=all_trace_file_names,
                                          random_seed=random_seed,
                                          video=Video(video_path, logdir='log'),
                                          #video=Video('bigbuckbunny2160p60.mp4', logdir='log'),
                                          writer = writer,
                                          test=test
                                          )

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.alpha = 0.02
        self.beta = -0.10
        self.gamma = -1.0
        self.delta = 0.01
        self.epsilon = 0.01
        self.B = []
        self.CRF = []
        self.state = np.zeros((S_INFO, S_LEN))

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time = 0
        self.time_stamp = 0
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        state = np.roll(self.state, -1, axis=1) # shift left ex. [1,2,3,4,5] -> [2,3,4,5,0]
        self.state = state
        return state

    def render(self):
        pass

    def close(self):
        pass

    def step(self, action):
        self.time += 1
        B, CRF = decoder_action(action)
        self.B.append(B)
        self.CRF.append(CRF)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
        video_chunk_size, end_of_video, pst, vmaf, last_vmaf, reward = self.net_env.get_video_chunk(B, CRF)
        print(f'Duration:{B/1000:.2f}s, CRF: {CRF}, Video_time_now: {self.net_env.video_time / 1000:.2f}s, Vmaf:{vmaf:.2f}, Pst: {pst:.2f}')
        # reward is video quality - smooth penalty - rebuffer penalty

        
        # get the new video feature state after taking the action
        video_features, network_condition_histogram, edge = self.net_env.get_next_features()

        # Extract the individual video features
        avg_si = video_features['Average SI']
        avg_ti = video_features['Average TI']
        avg_glcm = video_features['Average GLCM']
        avg_contrast = avg_glcm['contrast']
        avg_dissimilarity = avg_glcm['dissimilarity']
        avg_homogeneity = avg_glcm['homogeneity']
        avg_energy = avg_glcm['energy']
        avg_correlation = avg_glcm['correlation']
        avg_ASM = avg_glcm['ASM']

        # Shift the state array to the left to make room for the new features
        self.state = np.roll(self.state, -1, axis=1)
        Throughput = video_chunk_size* 8 / 1000 / (delay / M_IN_K) 
        # Update the state with the new features
        # State shape: (S_INFO, S_LEN) -> (10,10)
        self.state[0, -1] = delay / 1000.  # download time in seconds 
        self.state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR # Buffer size in seconds
        self.state[2, -1] = Throughput /1000 # Throughput in kbps
        self.state[3, -1] = self.B[self.time - 1] / 1000.
        self.state[4, -1] = self.CRF[self.time - 1] /51 # chunk CRF normalized
        self.state[5, -1] = avg_si / SI_TI_NORM_FACTOR
        self.state[6, -1] = avg_ti / SI_TI_NORM_FACTOR
        #print shape
        self.state[7,-1] = avg_contrast / AVG_CONTRAST_NORM_FACTOR
        self.state[8,-1] = avg_dissimilarity / AVG_DISSIMILARITY_NORM_FACTOR
        self.state[9,-1] = avg_homogeneity / AVG_HOMOGENEITY_NORM_FACTOR
        self.state[10,-1] = avg_energy / AVG_ENERGY_NORM_FACTOR
        self.state[11,-1] = avg_correlation / AVG_CORRELATION_NORM_FACTOR
        self.state[12,-1] = avg_ASM /  AVG_ASM_NORM_FACTOR
        #self.state[7:13, -1] = [avg_contrast, avg_dissimilarity, avg_homogeneity, avg_energy, avg_correlation, avg_ASM]
        #print('shape glacm',self.state[7, -6:].shape)
        #print('network_condition_histogram shape:', network_condition_histogram.shape)
        self.state[13:, :] = network_condition_histogram
        #print('edge shape:', edge.shape)
        #print('edge shape:', edge.reshape(-1, S_LEN).shape)
        self.state[14, :] = edge
        print(f'Each state component value: Delay{self.state[0, -1]:.2f}s, Buffer size: {self.state[1, -1]:.2f}, Throughput: {Throughput/1000:.2f}Mbps, B: {self.state[3, -1]:.2f}s, CRF: {self.state[4, -1]:.2f}, SI: {self.state[5, -1]:.2f}, TI: {self.state[6, -1]:.2f}')
        print(f'GLACM: Contrast: {self.state[7, -1]:.2f}, Dissimilarity: {self.state[8, -1]:.2f}, Homogeneity: {self.state[9, -1]:.2f}, Energy: {self.state[10, -1]:.2f}, Correlation: {self.state[11, -1]:.2f}, ASM: {self.state[12, -1]:.2f}')
        print(f'Nework_condition {self.state[13, -1]:.2f}, Edge: {self.state[14, -1]:.2f}')
        # Flatten the state array before returning it
        set_state = self.state
        # observation, reward, done, info = env.step(action)
        info = None
        return set_state, reward, end_of_video, info


def linear_mapping(value, in_min, in_max, out_min, out_max):
    # 线性映射公式
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def decoder_action(action):
    B, CRF = (action[0] + 1), (action[1] + 1)
    B = linear_mapping(B, 0, 2, 1000, 10000)
    CRF = round(linear_mapping(CRF, 0, 2, 15, 40)) # 0-51 too large -> 15-40 try
    return B, CRF


print(decoder_action([1, 1]))
