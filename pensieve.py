import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

from a3c import ActorNetwork, CriticNetwork, compute_gradients

# Pensieve 的參數
S_INFO = 6  # state dimension
S_LEN = 8  # history length
A_DIM = 6  # action dimension (quality levels)
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
RAND_RANGE = 1000
CHUNK_TIL_VIDEO_END_CAP = 48.0
REF_DUR = 4.0 

map_list = [0.025, 0.0625, 0.125, 0.2]

class Pensieve:
    def __init__(self, env, model_path='./pensieve_model/beta-1_normalized_109000.ckpt'):
        tf.reset_default_graph()

        self.env = env
        self.model_path = model_path
        
        # Initialize TensorFlow session
        self.sess = tf.Session()
        
        # Create actor and critic networks
        self.actor = ActorNetwork(self.sess,
                                 state_dim=[S_INFO, S_LEN], 
                                 action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        
        self.critic = CriticNetwork(self.sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # save neural net parameters

        self.state = np.zeros((S_INFO, S_LEN))
        
        # Load pretrained model
        if model_path is not None and tf.train.checkpoint_exists(model_path):
            self.saver.restore(self.sess, model_path)
            print("Pensieve model restored from:", model_path)
        else:
            print("Warning: No pretrained Pensieve model found at", model_path)
    
    def select_action(self, buffer_size, last_delay, last_bytes, sizes):
        # Prepare state
        self.state = np.roll(self.state, -1, axis=1)

        # Fill state information
        # [last quality, buffer size, throughput, delay, chunk sizes, remaining chunks]
        if len(self.env.data["QUALITY_INDEX"]) > 0:
            #last_quality = self.env.data["QUALITY_INDEX"][-1]
            #state[0, -1] = VIDEO_BIT_RATE[last_quality] / float(np.max(VIDEO_BIT_RATE))

            #last_bitrate_list = self.env.data["BITRATE_LIST"][-1]
            #self.state[0, -1] = self.env.data["BITRATE"][-1] / float(np.max(last_bitrate_list))
            self.state[0, -1] = map_list[self.env.data["QUALITY_INDEX"][-1]]
        else:
            self.state[0, -1] = 0.5  # default value
            
        self.state[1, -1] = buffer_size / 1000 / BUFFER_NORM_FACTOR
        
        if last_delay > 0:
            throughput = last_bytes / float(last_delay) / M_IN_K  # MB/s
            self.state[2, -1] = throughput / 10.0  # normalize
        else:
            self.state[2, -1] = 0
            
        self.state[3, -1] = float(last_delay) / M_IN_K / BUFFER_NORM_FACTOR
        
        # Normalize chunk sizes
        #state[4, :A_DIM] = np.array(sizes) / M_IN_K / M_IN_K / 10.0  # 10 MB
        padded_sizes = np.zeros(A_DIM)
        padded_sizes[:len(sizes)] = np.array(sizes)
        self.state[4, :A_DIM] = padded_sizes / M_IN_K / M_IN_K / 10.0
        # Remaining video chunks
        remaining_chunks = (self.env.total_video_time - self.env.video_time) / 1000 / REF_DUR
        self.state[5, -1] = np.minimum(remaining_chunks, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        
        # Get action probabilities
        action_prob = self.actor.predict(np.reshape(self.state, (1, S_INFO, S_LEN)))
        
        # mask
        mask = [1, 1, 1, 1, 0, 0]
        masked_prob = action_prob * mask
        masked_prob /= np.sum(masked_prob)

        # Select action (quality level)
        #action_cumsum = np.cumsum(masked_prob)
        #bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        bit_rate = np.argmax(masked_prob)
         # === DEBUG PRINT START ===
        print("=== Pensieve Debug ===")
        print("state shape:", self.state.shape)
        print("state:\n", self.state)
        print("action_prob:", action_prob)
        print("mask:", mask)
        print("masked_prob:", masked_prob)
        print("selected bit_rate:", bit_rate)
        print("available sizes:", sizes)
        print("=======================")
        # === DEBUG PRINT END ===
        return bit_rate
    
    def train(self, s_batch, a_batch, r_batch, terminal):
        actor_gradients, critic_gradients, td_batch = compute_gradients(
            s_batch, a_batch, r_batch, terminal, self.actor, self.critic)
        
        self.actor.apply_gradients(actor_gradients)
        self.critic.apply_gradients(critic_gradients)
        
        return td_batch
    
    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Model saved in file: %s" % save_path)