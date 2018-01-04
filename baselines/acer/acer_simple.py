import time
import os
import joblib
import numpy as np
import gc
import tensorflow as tf
import pickle
from baselines import logger
from baselines.common import set_global_seeds
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance
from baselines.acer.buffer import Buffer
from baselines.acer.expert import Expert


# remove last step
def strip(var, nenvs, nsteps, flat = False):
	vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
	return seq_to_batch(vars[:-1], flat)

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
	"""
	Calculates q_retrace targets

	:param R: Rewards
	:param D: Dones
	:param q_i: Q values for actions taken
	:param v: V values
	:param rho_i: Importance weight for each action
	:return: Q_retrace values
	"""
	rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
	rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
	ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
	q_is = batch_to_seq(q_i, nenvs, nsteps, True)
	vs = batch_to_seq(v, nenvs, nsteps + 1, True)
	v_final = vs[-1]
	qret = v_final
	qrets = []
	for i in range(nsteps - 1, -1, -1):
		check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
		qret = rs[i] + gamma * qret * (1.0 - ds[i])
		qrets.append(qret)
		qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
	qrets = qrets[::-1]
	qret = seq_to_batch(qrets, flat=True)
	return qret

# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))

class Model(object):
	def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
				 ent_coef, q_coef, gamma, max_grad_norm, lr,
				 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
				 c, trust_region, alpha, delta, network_saving_dir, use_expert, expert):
		config = tf.ConfigProto(allow_soft_placement=True,
								intra_op_parallelism_threads=num_procs,
								inter_op_parallelism_threads=num_procs)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		nact = ac_space.n
		nbatch = nenvs * nsteps
  
		A = tf.placeholder(tf.int32, [nbatch]) # actions
		D = tf.placeholder(tf.float32, [nbatch]) # dones
		R = tf.placeholder(tf.float32, [nbatch]) # rewards, not returns
		MU = tf.placeholder(tf.float32, [nbatch, nact]) # mu's
		LR = tf.placeholder(tf.float32, [])
		eps = 1e-6

		#USE_EXP = tf.placeholder(tf.bool)

		step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
		train_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)
		expert_train_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)


		params = find_trainable_variables("model")
		print("Params {}".format(len(params)))
		for var in params:
			print(var)

		# create polyak averaged model
		ema = tf.train.ExponentialMovingAverage(alpha)
		ema_apply_op = ema.apply(params)

		def custom_getter(getter, *args, **kwargs):
			v = ema.average(getter(*args, **kwargs))
			print(v.name)
			return v

		with tf.variable_scope("", custom_getter=custom_getter, reuse=True):
			polyak_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)
			#expert_polyak_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)


		# Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i
		v = tf.reduce_sum(train_model.pi * train_model.q, axis = -1) # shape is [nenvs * (nsteps + 1)]

		# strip off last step
		f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model.pi, polyak_model.pi, train_model.q])
		# Get pi and q values for actions taken
		f_i = get_by_index(f, A)
		q_i = get_by_index(q, A)

		# Compute ratios for importance truncation
		rho = f / (MU + eps)
		rho_i = get_by_index(rho, A)

		# Calculate Q_retrace targets
		qret = q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma)

		# Calculate losses
		# Entropy
		entropy = tf.reduce_mean(cat_entropy_softmax(f))



		# Policy Gradient loss, with truncated importance sampling & bias correction
		v = strip(v, nenvs, nsteps, True)
		check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
		check_shape([rho, f, q], [[nenvs * nsteps, nact]] * 2)

		# Truncated importance sampling
		adv = qret - v
		logf = tf.log(f_i + eps)
		gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
		loss_f = -tf.reduce_mean(gain_f)

		if use_expert:
			expert.set_tf(sess, expert_train_model, ob_space, ac_space, nenvs, nsteps)

		# Bias correction for the truncation
		adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
		logf_bc = tf.log(f + eps) # / (f_old + eps)
		check_shape([adv_bc, logf_bc], [[nenvs * nsteps, nact]]*2)
		gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f), axis = 1) #IMP: This is sum, as expectation wrt f
		loss_bc= -tf.reduce_mean(gain_bc)

		loss_policy = loss_f + loss_bc

		# Value/Q function loss, and explained variance
		check_shape([qret, q_i], [[nenvs * nsteps]]*2)
		ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))

		if use_expert:
			expert_loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i) * 0.5) + expert.expert_loss
			#expert_loss_q = expert.loss_q
		else:
			expert_loss_q =None


		loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i) * 0.5)

		# Net loss
		check_shape([loss_policy, loss_q, entropy], [[]] * 3)
		loss = loss_policy + q_coef * loss_q - ent_coef * entropy

		if use_expert:
			expert_loss = loss_policy + q_coef * expert.expert_loss - ent_coef * entropy
			#expert_loss = loss_policy + q_coef * expert_loss_q - ent_coef * entropy + expert.loss_policy
		else:
			expert_loss = None


		if trust_region:
			g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f) #[nenvs * nsteps, nact]
			# k = tf.gradients(KL(f_pol || f), f)
			k = - f_pol / (f + eps) #[nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
			k_dot_g = tf.reduce_sum(k * g, axis=-1)
			adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) / (tf.reduce_sum(tf.square(k), axis=-1) + eps)) #[nenvs * nsteps]

			# Calculate stats (before doing adjustment) for logging.
			avg_norm_k = avg_norm(k)
			avg_norm_g = avg_norm(g)
			avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
			avg_norm_adj = tf.reduce_mean(tf.abs(adj))

			g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k
			grads_f = -g/(nenvs*nsteps) # These are turst region adjusted gradients wrt f ie statistics of policy pi
			grads_policy = tf.gradients(f, params, grads_f)
			grads_q = tf.gradients(loss_q * q_coef, params)
			grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]
			avg_norm_grads_f = avg_norm(grads_f) * (nsteps * nenvs)
			norm_grads_q = tf.global_norm(grads_q)
			norm_grads_policy = tf.global_norm(grads_policy)

			if use_expert:
				expert_grads_q = tf.gradients(expert_loss_q * q_coef, params)
				expert_grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, expert_grads_q, params)]

			else:
				expert_grads = None


		else:
			grads = tf.gradients(loss, params)
			if use_expert:
				expert_grads = tf.gradients(expert_loss, params)
			else:
				expert_grads = None

		if max_grad_norm is not None:
			grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
			if use_expert:
				expert_grads, expert_norm_grads = tf.clip_by_global_norm(expert_grads, max_grad_norm)


		grads = list(zip(grads, params))
		trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
		_opt_op = trainer.apply_gradients(grads)

		if use_expert:
			expert_grads = list(zip(expert_grads, params))
			expert_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
			expert_opt_op = expert_trainer.apply_gradients(expert_grads)


		# so when you call _train, you first do the gradient step, then you apply ema
		with tf.control_dependencies([_opt_op]):
			_train = tf.group(ema_apply_op)

		if use_expert:
			with tf.control_dependencies([expert_opt_op]):
				expert_train = tf.group(ema_apply_op)


		lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

		# Ops/Summaries to run, and their names for logging
		if use_expert:
			expert_run_ops = [expert_train, expert_loss, expert.expert_loss, entropy, loss_policy, loss_f, loss_bc, ev,
							  norm_grads, expert.loss_q, expert.loss_policy]
			expert_names_ops = ['loss', 'expert_loss', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
						 'norm_grads', 'expert_loss_q', 'expert_loss_policy']
		else:
			expert_run_ops = None
			expert_names_ops = None

		run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
		names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
						 'norm_grads']


		if trust_region:
			run_ops = run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
								 avg_norm_adj]
			names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g',
									 'avg_norm_k_dot_g', 'avg_norm_adj']

			if use_expert:
				expert_run_ops = expert_run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g,
									 avg_norm_k_dot_g,
									 avg_norm_adj]
				expert_names_ops = expert_names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k',
										 'avg_norm_g',
										 'avg_norm_k_dot_g', 'avg_norm_adj']
			else:
				expert_run_ops = None
				expert_names_ops = None

		
		


		def train(obs, actions, rewards, dones, mus, states, masks, steps):
			cur_lr = lr.value_steps(steps)
			td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
			if states != []:
				td_map[train_model.S] = states
				td_map[train_model.M] = masks
				td_map[polyak_model.S] = states
				td_map[polyak_model.M] = masks

			return names_ops, sess.run(run_ops, td_map)[1:]  # strip off _train

		def expert_train(obs, actions, rewards, dones, mus, states, masks, steps,
						 expert_obs, expert_actions, expert_rewards, expert_mus, expert_dones, expert_masks):
			cur_lr = lr.value_steps(steps)
			td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr,
					  expert.A:expert_actions,expert.R:expert_rewards, expert.D:expert_dones, expert.MU:expert_mus,
					  expert_train_model.X:expert_obs}
			if states != []:
				td_map[train_model.S] = states
				td_map[train_model.M] = masks
				td_map[expert_train_model.S] = states
				td_map[expert_train_model.M] = expert_masks
				td_map[polyak_model.S] = states
				td_map[polyak_model.M] = masks


			return expert_names_ops, sess.run(expert_run_ops, td_map)[1:]  # strip off _train


		# def save(save_path):
		#     ps = sess.run(params)
		#     make_path(save_path)
		#     joblib.dump(ps, save_path)

		def save(training_step):
			logger.info('Saved network with {} training steps'.format(training_step))
			self.saver.save(self.sess, self.cpk_dir + 'demo', global_step=training_step)
		
		def load():
			print(self.cpk_dir)
			checkpoint = tf.train.get_checkpoint_state(self.cpk_dir)
			if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
				logger.info('Successfully loaded {}'.format(checkpoint.model_checkpoint_path))
			else:
				logger.info('Could not find old network weights')

		saver = tf.train.Saver()
		self.saver = saver
		self.sess = sess
		self.cpk_dir = network_saving_dir
		self.save = save
		self.load = load

		self.train = train
		self.expert_train = expert_train
		self.train_model = train_model
		self.step_model = step_model
		self.step = step_model.step
		self.initial_state = step_model.initial_state
		tf.global_variables_initializer().run(session=sess)

class Runner(object):
	def __init__(self, env, model, nsteps, nstack):
		self.env = env
		self.nstack = nstack
		self.model = model
		nh, nw, nc = env.observation_space.shape
		self.nc = nc  # nc = 1 for atari, but just in case
		self.nenv = nenv = env.num_envs
		self.nact = env.action_space.n
		self.nbatch = nenv * nsteps
		self.batch_ob_shape = (nenv*(nsteps+1), nh, nw, nc*nstack)
		self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
		obs = env.reset()
		self.update_obs(obs)
		self.nsteps = nsteps
		self.states = model.initial_state
		self.dones = [False for _ in range(nenv)]

	def update_obs(self, obs, dones=None):
		if dones is not None:
			self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
		self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
		self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]

	def run(self):
		enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
		mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
		for _ in range(self.nsteps):
			actions, mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)
			mb_obs.append(np.copy(self.obs))
			mb_actions.append(actions)
			mb_mus.append(mus)
			mb_dones.append(self.dones)
			obs, rewards, dones, _ = self.env.step(actions)
			# states information for statefull models like LSTM
			self.states = states
			self.dones = dones
			self.update_obs(obs, dones)
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

		# shapes are now [nenv, nsteps, []]
		# When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

		return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks

class Acer():
	def __init__(self, runner, model, buffer, log_interval, expert_buffer = None):
		self.runner = runner
		self.model = model
		self.buffer = buffer
		self.log_interval = log_interval
		self.tstart = None
		self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
		self.steps = None
		self.expert_buffer = []


		#self.flag = 1

	def call(self, perform, save_networks, use_expert,expert, on_policy):
		runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps
		expert_buffer = self.expert_buffer

		if on_policy:
			enc_obs, obs, actions, rewards, mus, dones, masks = runner.run()
			# if self.flag>0:
			# 	print(self.flag,'=================================')
			# 	print(enc_obs, obs, actions, rewards, mus, dones, masks)
			# 	self.flag = self.flag -1
			self.episode_stats.feed(rewards, dones)
			if buffer is not None and not perform:
				buffer.put(enc_obs, actions, rewards, mus, dones, masks)
		else:
			# get obs, actions, rewards, mus, dones from buffer.
			obs, actions, rewards, mus, dones, masks = buffer.get()



		if not perform: 
		# reshape stuff correctly
			obs = obs.reshape(runner.batch_ob_shape)
			actions = actions.reshape([runner.nbatch])
			rewards = rewards.reshape([runner.nbatch])
			mus = mus.reshape([runner.nbatch, runner.nact])
			dones = dones.reshape([runner.nbatch])
			masks = masks.reshape([runner.batch_ob_shape[0]])

			if not use_expert:
				names_ops, values_ops = model.train(obs, actions, rewards, dones, mus, model.initial_state, masks, steps)
			else:
				expert_obs, expert_actions, expert_rewards, expert_mus, expert_dones, expert_masks = expert.get()
				expert_obs = expert_obs.reshape(runner.batch_ob_shape)
				expert_actions = expert_actions.reshape([runner.nbatch])
				expert_rewards = expert_rewards.reshape([runner.nbatch])
				expert_mus = expert_mus.reshape([runner.nbatch, runner.nact])
				expert_dones = expert_dones.reshape([runner.nbatch])
				expert_masks = expert_masks.reshape([runner.batch_ob_shape[0]])
				names_ops, values_ops = model.expert_train(obs, actions, rewards, dones, mus, model.initial_state, masks,
													steps,  expert_obs, expert_actions, expert_rewards, expert_mus,
													expert_dones, expert_masks)


			if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
				logger.record_tabular("total_timesteps", steps)
				logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
				# IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
				# Thus, this is mean until end of life, not end of episode.
				# For true episode rewards, see the monitor files in the log folder.
				logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
				logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())
				for name, val in zip(names_ops, values_ops):
					logger.record_tabular(name, float(val))
				logger.dump_tabular()

				if save_networks and (int(steps/runner.nbatch) % self.log_interval*10 == 0):
					model.save(int(steps))

		else:  #if perform
			expert_buffer.append([enc_obs, actions, rewards, mus, dones, masks])
			# print('***********************')
			# print(rewards)
			# print('***********************')

			if len(expert_buffer)>0 and len(expert_buffer)%100 == 0:
				expert_dir = os.path.join('./expert')+'/'
				if not os.path.exists(expert_dir):
					os.makedirs(expert_dir)
				pwritefile = open(os.path.join(expert_dir, 'expert.pkl'),'wb')
				pickle.dump(expert_buffer,pwritefile,-1)
				pwritefile.close()
				logger.info('Successfully Saved the Expert Data')

			obs = obs.reshape(runner.batch_ob_shape)
			actions = actions.reshape([runner.nbatch])
			rewards = rewards.reshape([runner.nbatch])
			mus = mus.reshape([runner.nbatch, runner.nact])
			dones = dones.reshape([runner.nbatch])
			masks = masks.reshape([runner.batch_ob_shape[0]])
			if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
				logger.record_tabular("total_timesteps", steps)
				logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
				# IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
				# Thus, this is mean until end of life, not end of episode.
				# For true episode rewards, see the monitor files in the log folder.
				logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
				logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())

				logger.dump_tabular()


def learn(policy, env, seed, perform = False, use_expert = False, save_networks = False, network_saving_dir = None, total_timesteps = int(80e6),nsteps=20,nstack=4, q_coef=0.5, ent_coef=0.01,
		  max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
		  log_interval=10, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0,
		  trust_region=True, alpha=0.99, delta=1):

	print(locals())
	tf.reset_default_graph()
	set_global_seeds(seed)

	nenvs = env.num_envs
	ob_space = env.observation_space   #Box(84,84,1)
	ac_space = env.action_space   #Discrete(4)

	if use_expert:
		expert = Expert(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
		expert_dir = os.path.join('./expert') + '/expert.pkl'
		expert.load_file(expert_dir)
	else:
		expert = None
	

	num_procs = len(env.remotes) # HACK
	model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
				  num_procs=num_procs, ent_coef=ent_coef, q_coef=q_coef, gamma=gamma,
				  max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon,
				  total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
				  trust_region=trust_region, alpha=alpha, delta=delta, network_saving_dir = network_saving_dir, use_expert=use_expert, expert = expert)

	runner = Runner(env=env, model=model, nsteps=nsteps, nstack=nstack)

	if replay_ratio > 0:
		buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
	else:
		buffer = None

	if perform:
		model.load()

	nbatch = nenvs*nsteps
	acer = Acer(runner, model, buffer, log_interval)
	acer.tstart = time.time()

	for acer.steps in range(0, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
		

		# if acer.steps >5e4 and use_expert:
		# 	print('-------------------------')
		# 	print('Reuse the normal networks')
		# 	print('-------------------------')
		# 	use_expert = False
		# 	expert = None

		acer.call(perform, save_networks, use_expert, expert, on_policy=True)

		if replay_ratio > 0 and buffer.has_atleast(replay_start) and not perform:
			n = np.random.poisson(replay_ratio)
			for _ in range(n):
				acer.call(perform, save_networks,use_expert, expert,on_policy=False)  # no simulation steps in this

	#dir = os.path.join('./models/', 'test.m')
	#model.save('./models/test_2.pkl')
	#
	#


	env.close()
