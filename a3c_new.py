import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 4

class ActorNetwork(Model):
    def __init__(self, state_dim, action_dim, learning_rate):
        super(ActorNetwork, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        
        # 构建网络
        self.split_0 = layers.Dense(128, activation='relu')
        self.split_1 = layers.Dense(128, activation='relu')
        self.split_2 = layers.Conv1D(128, 4, activation='relu')
        self.split_3 = layers.Conv1D(128, 4, activation='relu')
        self.split_4 = layers.Conv1D(128, 4, activation='relu')
        self.split_5 = layers.Dense(128, activation='relu')
        
        self.flatten = layers.Flatten()
        self.dense_0 = layers.Dense(128, activation='relu')
        self.out_layer = layers.Dense(action_dim, activation='softmax')
        
        # 优化器
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
    def call(self, inputs):
        # 处理不同部分的输入
        split_0 = self.split_0(inputs[:, 0:1, -1])
        split_1 = self.split_1(inputs[:, 1:2, -1])
        split_2 = self.split_2(inputs[:, 2:3, :])
        split_3 = self.split_3(inputs[:, 3:4, :])
        split_4 = self.split_4(inputs[:, 4:5, :A_DIM])
        split_5 = self.split_5(inputs[:, 4:5, -1])
        
        # 展平卷积输出
        split_2_flat = self.flatten(split_2)
        split_3_flat = self.flatten(split_3)
        split_4_flat = self.flatten(split_4)
        
        # 合并所有特征
        merge_net = tf.concat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)
        
        # 全连接层
        dense_net = self.dense_0(merge_net)
        out = self.out_layer(dense_net)
        
        return out
    
    def train_step(self, inputs, acts, act_grad_weights):
        with tf.GradientTape() as tape:
            # 前向传播
            out = self(inputs, training=True)
            
            # 计算损失
            policy_loss = tf.reduce_sum(
                tf.multiply(
                    tf.math.log(tf.reduce_sum(tf.multiply(out, acts), axis=1, keepdims=True)),
                    -act_grad_weights
                )
            )
            entropy_loss = ENTROPY_WEIGHT * tf.reduce_sum(
                tf.multiply(out, tf.math.log(out + ENTROPY_EPS)))
            total_loss = policy_loss + entropy_loss
        # 计算梯度并更新权重
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return gradients

class CriticNetwork(Model):
    def __init__(self, state_dim, learning_rate):
        super(CriticNetwork, self).__init__()
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        
        # 构建网络 (结构与Actor类似)
        self.split_0 = layers.Dense(128, activation='relu')
        self.split_1 = layers.Dense(128, activation='relu')
        self.split_2 = layers.Conv1D(128, 4, activation='relu')
        self.split_3 = layers.Conv1D(128, 4, activation='relu')
        self.split_4 = layers.Conv1D(128, 4, activation='relu')
        self.split_5 = layers.Dense(128, activation='relu')
        
        self.flatten = layers.Flatten()
        self.dense_0 = layers.Dense(128, activation='relu')
        self.out_layer = layers.Dense(1, activation='linear')
        
        # 优化器
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
    def call(self, inputs):
        # 处理输入 (与Actor相同)
        split_0 = self.split_0(inputs[:, 0:1, -1])
        split_1 = self.split_1(inputs[:, 1:2, -1])
        split_2 = self.split_2(inputs[:, 2:3, :])
        split_3 = self.split_3(inputs[:, 3:4, :])
        split_4 = self.split_4(inputs[:, 4:5, :A_DIM])
        split_5 = self.split_5(inputs[:, 4:5, -1])
        
        split_2_flat = self.flatten(split_2)
        split_3_flat = self.flatten(split_3)
        split_4_flat = self.flatten(split_4)
        
        merge_net = tf.concat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)
        
        dense_net = self.dense_0(merge_net)
        out = self.out_layer(dense_net)
        
        return out
    
    def train_step(self, inputs, td_target):
        with tf.GradientTape() as tape:
            # 前向传播
            out = self(inputs, training=True)
            
            # 计算损失 (MSE)
            loss = tf.reduce_mean(tf.square(td_target - out))
        
        # 计算梯度并更新权重
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, gradients

def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic(s_batch).numpy()

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    # 计算actor梯度
    with tf.GradientTape() as tape:
        out = actor(s_batch, training=True)
        policy_loss = tf.reduce_sum(
            tf.multiply(
                tf.math.log(tf.reduce_sum(tf.multiply(out, a_batch), axis=1, keepdims=True)),
                -td_batch
            )
        )
        entropy_loss = ENTROPY_WEIGHT * tf.reduce_sum(
            tf.multiply(out, tf.math.log(out + ENTROPY_EPS)))
        actor_loss = policy_loss + entropy_loss
    
    actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
    
    # 计算critic梯度
    with tf.GradientTape() as tape:
        out = critic(s_batch, training=True)
        critic_loss = tf.reduce_mean(tf.square(R_batch - out))
    
    critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)

    return actor_gradients, critic_gradients, td_batch

# 其余辅助函数保持不变
def discount(x, gamma):
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    return out

def compute_entropy(x):
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars