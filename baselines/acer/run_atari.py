#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds,boolean_flag
from baselines import bench
from baselines.acer.acer_simple import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
import datetime
import argparse


def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, perform, use_expert, save_networks):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return

    network_saving_dir = os.path.join('./saved_networks', env_id)+'/'
    if not os.path.exists(network_saving_dir):
        os.makedirs(network_saving_dir)
        
    learn(policy_fn, env, seed, perform, use_expert, save_networks, network_saving_dir, int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--logdir', help ='Directory for logging', default='./log')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--log-dir', type=str, default=None)
    boolean_flag(parser, 'perform', default=False)
    boolean_flag(parser, 'use-expert', default=False)
    boolean_flag(parser, 'save-networks', default=False)

    args = parser.parse_args()

    # logger.configure(os.path.abspath(args.logdir))

    # dir = os.path.join('./logs/', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    # logger.configure(dir=dir)

    #set the log_dir

    if args.log_dir is None:
        dir = os.path.join('./logs/', args.env,
                           datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    else:
        dir = os.path.join('./logs/', args.env, args.log_dir)
    logger.configure(dir=dir)

    del args.log_dir


    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16, perform = args.perform, use_expert = args.use_expert, save_networks = args.save_networks)

if __name__ == '__main__':
    main()
