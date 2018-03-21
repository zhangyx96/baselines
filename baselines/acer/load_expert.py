import numpy as np
import pickle
import os

file_dir = os.path.join('./expert','expert.pkl')
expert_file = open(file_dir, 'rb')
expert_data = pickle.load(expert_file)
expert_file.close()
sum = np.zeros(16)


for step_sample in expert_data:
    #(step_sample[0], step_sample[1], step_sample[2], step_sample[3], step_sample[4], step_sample[5])
    #print(step_sample[2].shape[0],step_sample[2].shape[1])

    for x in range(step_sample[2].shape[0]):
        for y in range(step_sample[2].shape[1]):
            sum[x] += step_sample[2][x][y]
            if step_sample[5][x][y]:
                print('sum[',x,'] = ',sum[x])
                sum[x] = 0





del expert_data
