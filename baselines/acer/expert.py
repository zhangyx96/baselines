import pickle
import tensorflow as tf
import gc
from baselines.acer.buffer import Buffer

from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import check_shape, avg_norm, gradient_add, q_explained_variance, get_by_index
from baselines.a2c.utils import batch_to_seq, seq_to_batch

def strip(var, nenvs, nsteps, flat = False):
	vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
	return seq_to_batch(vars[:-1], flat)

class Expert:
	def __init__(self, env, nsteps, nstack, size):
		self.env = env
		self.nsteps = nsteps
		self.nstack = nstack
		self.size = size
		self.buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size= size)
		self.file_dir = None
		self.flag = 3

	def load_file(self, file_dir):
		self.file_dir = file_dir
		expert_file = open(self.file_dir, 'rb')
		expert_data = pickle.load(expert_file)
		expert_file.close()
		for step_sample in expert_data:
			self.buffer.put(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
			if self.flag > 0:
				print(self.flag,'**************************************')
				print(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
				self.flag = self.flag -1
		del expert_data

		gc.collect()

	def get(self):
		return self.buffer.get()

	def strip(var, nenvs, nsteps, flat=False):
		vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
		return seq_to_batch(vars[:-1], flat)

	def set_tf(self, sess, expert_train_model, ob_space, ac_space, nenvs, nsteps):
		nact = ac_space.n
		nbatch = nenvs * nsteps
		self.A = tf.placeholder(tf.int32, [nbatch])  # actions
		self.D = tf.placeholder(tf.float32, [nbatch])  # dones
		self.R = tf.placeholder(tf.float32, [nbatch])  # rewards, not returns
		self.MU = tf.placeholder(tf.float32, [nbatch, nact])  # mu's
		self.LR = tf.placeholder(tf.float32, [])
		eps = 1e-6

		#step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)


		# params = find_trainable_variables("model")
		# print("Params {}".format(len(params)))
		# for var in params:
		# 	print(var)

		# create polyak averaged model
		#ema = tf.train.ExponentialMovingAverage(alpha)
		#ema_apply_op = ema.apply(params)


		# Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

		v = tf.reduce_sum(tf.stop_gradient(expert_train_model.pi) * expert_train_model.q, axis = -1) # shape is [nenvs * (nsteps + 1)]
		s_v = tf.reduce_sum(expert_train_model.pi * tf.stop_gradient(expert_train_model.q), axis = -1)
		v = strip(v, nenvs, nsteps, True)
		s_v = strip(s_v, nenvs, nsteps, True)
		# strip off last step
		#f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [expert_train_model.pi, expert_polyak_model.pi, expert_train_model.q])

		fq = lambda var: strip(var, nenvs, nsteps)

		q_i = get_by_index(fq(expert_train_model.q),self.A)
		#v = tf.reduce_max(fq(expert_train_model.q), axis = 1)

		# one_hot_A = tf.one_hot(self.A, nact)
		# pi = fq(expert_train_model.pi)
		# loss_policy = tf.reduce_mean(tf.square(pi-one_hot_A))


		# Get pi and q values for actions taken

		#v = strip(v, nenvs, nsteps, True)


		#loss_q = -tf.reduce_mean(q_i - tf.reshape(v, [nenvs * nsteps, 1]))
		loss_q = tf.nn.relu(tf.reduce_mean(v - q_i))
		loss_policy = -tf.reduce_mean(s_v - tf.stop_gradient(q_i))
		self.expert_loss = loss_q+loss_policy
		self.loss_q = loss_q
		self.loss_policy = loss_policy







