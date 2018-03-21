import pickle
import tensorflow as tf
import gc
import os
from baselines.acer.buffer import Buffer
import numpy as np
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
			# print('----------')
			# print(step_sample[5].shape)
			# print('----------')
			self.buffer.put(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
			# if self.flag > 0:
			# 	print(self.flag,'**************************************')
			# 	print(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
			# 	self.flag = self.flag -1
		del expert_data
		gc.collect()


	def load_file_human(self,file_dir = '/home/zhangxiaoqin/Projects/conda/atari_v1/'):
		import agc.dataset as ds
		import agc.util as util
		import cv2
		env_name = 'spaceinvaders'
		nsteps = 20
		next_file_point = 1
		file_point = np.arange(16,dtype = np.int)

		frame_point = np.zeros((16),dtype = np.int)  #f_p[0][0]   first_line->file_num,  sec_line->frame_num
		dataset = ds.AtariDataset(file_dir)
		all_trajectories = dataset.trajectories
		num = all_trajectories
		screenshoot_dir = os.path.join(file_dir,'screens/spaceinvaders')
		flag = 1
		k=0
		while k<16:
			i=1
			if i in dataset.trajectories['spaceinvaders']:
				file_point[k]=i
				k = k+1
			i = i+1

		init_obs = np.zeros((16, 84, 84, 4), dtype=np.uint8)
		enc_obs = np.split(init_obs, 4, axis=3)  # so now list of obs steps
		mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
		for _ in range(nsteps):
			#actions, mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)
			obs = np.zeros([16,84,84,1],dtype = np.uint8)
			while(flag):
				print('----------------------')
				print(file_point)
				print(frame_point)
				#print(next_file_point)
				for i in np.arange(16):
					pic_path = os.path.join(screenshoot_dir,str(file_point[i]),str(frame_point[i]))+'.png'
					pic = cv2.imread(pic_path)
					pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
					pic = cv2.resize(pic, (84, 84), interpolation=cv2.INTER_AREA)
					obs[i,:,:,:] = pic[:,:,None]
					if frame_point[i] < all_trajectories['spaceinvaders'][file_point[i]][-1]['frame']:
						frame_point[i] = frame_point[i] + 1
					else:
						frame_point[i] = 0
						file_point[i] = next_file_point
						next_file_point = next_file_point + 1

						while next_file_point not in dataset.trajectories['spaceinvaders'] and next_file_point<=514:
							next_file_point = next_file_point+1
					if next_file_point>514:
						flag = False
			mb_obs.append(np.copy(self.obs))
			mb_actions.append(actions)
			mb_mus.append(mus)
			mb_dones.append(self.dones)
			#obs, rewards, dones, _ = self.env.step(actions)
			#env.render();
			# aa,bb,cc,dd = self.env_s.step(actions[0])
			# self.env_s.render()
			# if cc == True:
			# 	self.env_s.reset()
			# states information for statefull models like LSTM
			self.states = states
			self.dones = dones
			#self.update_obs(obs, dones)
			mb_rewards.append(rewards)
			enc_obs.append(obs)
		mb_obs.append(np.copy(self.obs))
		mb_dones.append(self.dones)
		enc_obs = np.asarray(enc_obs, dtype=np.uint8).swapaxes(1, 0)
		mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
		mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
		mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
		mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
		mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
		mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards






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
		#self.expert_loss = loss_policy
		self.loss_q = loss_q
		self.loss_policy = loss_policy
