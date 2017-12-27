import pickle
import tensorflow as tf
from baselines.acer.buffer import Buffer

class Expert:
	def __init__(self, env, nsteps, nstack, size):
		self.env = env
		self.nsteps = nsteps
		self.nstack = nstack
		self.size = size
		self.buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size= size)
		self.file_dir = None
		#self.flag = 1

	def load_file(self, file_dir):
		self.file_dir = file_dir
		expert_file = open(self.file_dir, 'rb')
		expert_data = pickle.load(expert_file)
		expert_file.close()
		for step_sample in expert_data:
			self.buffer.put(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
			# if self.flag > 0:
			# 	print(self.flag,'**************************************')
			# 	print(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
			# 	self.flag = self.flag -1
	def sample:
		return self.buffer.get()
