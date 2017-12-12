import pickle
import tensorflow as tf
from baselines.ddpg.memory import Memory
from baselines.ddpg.ddpg import normalize, denormalize


class Expert:
    def __init__(self, limit, env):
        self.limit = limit
        self.env = env
        self.memory = Memory(limit=self.limit,
                             action_shape=self.env.action_space.shape,
                             observation_shape=self.env.observation_space.shape)
        self.file_dir = None
        self.expert_state = tf.placeholder(tf.float32, shape=(None,)+self.env.observation_space.shape, name='expert_state')
        self.expert_action = tf.placeholder(tf.float32, shape=(None,)+self.env.action_space.shape, name='expert_action')
        self.Q_with_expert_data = None
        self.Q_with_expert_actor = None
        self.loss = None
        self.obs_rms = None

    def load_file(self, file_dir):
        self.file_dir = file_dir
        expert_file = open(self.file_dir, 'rb')
        expert_data = pickle.load(expert_file)
        expert_file.close()
        for episode_sample in expert_data:
            for step_sample in episode_sample:
                self.memory.append(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4])

    def sample(self, batch_size):
        return self.memory.sample(batch_size)

    def set_tf(self, actor, critic, obs_rms, ret_rms, observation_range, return_range):

        normalized_state = tf.clip_by_value(normalize(self.expert_state, obs_rms),
                                            observation_range[0], observation_range[1])
        expert_actor = actor(normalized_state, reuse=True)
        normalized_q_with_expert_data = critic(normalized_state, self.expert_action, reuse=True)
        normalized_q_with_expert_actor = critic(normalized_state, expert_actor, reuse=True)
        self.Q_with_expert_data = denormalize(
            tf.clip_by_value(normalized_q_with_expert_data, return_range[0], return_range[1]), ret_rms)
        self.Q_with_expert_actor = denormalize(
            tf.clip_by_value(normalized_q_with_expert_actor, return_range[0], return_range[1]), ret_rms)
        self.loss = tf.reduce_mean(self.Q_with_expert_actor - self.Q_with_expert_data)
