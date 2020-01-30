import numpy as np
import pickle
from datetime import datetime

import sys
import os
sys.path.append('../')

# local imports
import envs

import torch
from sac import SoftActorCritic
from sac import PolicyNetwork
from sac import ReplayBuffer
from sac import NormalizedActions
from hybrid_stochastic import PathIntegral
from model import ModelOptimizer, Model, SARSAReplayBuffer
from model import MDNModelOptimizer, MDNModel
# argparse things
import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-4)
parser.add_argument('--value_lr',   type=float, default=3e-4)
parser.add_argument('--soft_q_lr',  type=float, default=3e-4)

parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--model_iter', type=int, default=2)
parser.add_argument('--trajectory_samples', type=int, default=20)
parser.add_argument('--lam',  type=float, default=1.0)


parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)

args = parser.parse_args()

class ActionWrapper(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.mean = np.mean([action_space.low, action_space.high], axis=0)
        self.std = np.std([action_space.low, action_space.high], axis=0)

    def __call__(self, action):
        return action
        # return (action + self.mean)*self.std



if __name__ == '__main__':

    # env_name = 'KukaBulletEnv-v0'
    # env_name = 'InvertedPendulumSwingupBulletEnv-v0'
    # env_name = 'ReacherBulletEnv-v0'
    # env_name = 'HalfCheetahBulletEnv-v0'
    # env = pybullet_envs.make(env_name)
    # env.isRender = True
    # env = KukaGymEnv(renders=True, isDiscrete=False)


    env_name = args.env
    try:
        env = envs.env_list[env_name](render=args.render)
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = envs.env_list[env_name]()
    env.reset()
    print(env.action_space.low, env.action_space.high)
    action_wrapper = ActionWrapper(env.action_space)
    # assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    print('CHECKING THE WRAPPER')
    # print(action_wrapper(np.ones(env.action_space.shape)))
    # print(action_wrapper(-np.ones(env.action_space.shape)))

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")

    path = './data/' + env_name +  '/' + 'h_sac/' + date_str
    if os.path.exists(path) is False:
        os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    model = Model(state_dim, action_dim, def_layers=[256])
    # model = MDNModel(state_dim, action_dim, def_layers=[200, 200])


    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
    model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)

    # model_optim = MDNModelOptimizer(model, replay_buffer, lr=args.model_lr)


    sac = SoftActorCritic(policy=policy_net,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          replay_buffer=replay_buffer,
                          policy_lr=args.policy_lr,
                          value_lr=args.value_lr,
                          soft_q_lr=args.soft_q_lr)

    planner = PathIntegral(model, policy_net, samples=args.trajectory_samples, t_H=args.horizon, lam=args.lam)

    max_frames  = args.max_frames
    max_steps   = args.max_steps
    frame_skip = args.frame_skip

    frame_idx   = 0
    rewards     = []
    batch_size  = 128

    # env.camera_adjust()
    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        planner.reset()

        action = planner(state)

        episode_reward = 0
        for step in range(max_steps):
            # action = policy_net.get_action(state)
            for _ in range(frame_skip):
                next_state, reward, done, info = env.step(action_wrapper(action.copy()))

            next_action = planner(next_state)

            replay_buffer.push(state, action, reward, next_state, done)
            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            if len(replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)
                model_optim.update_model(batch_size, mini_iter=args.model_iter)
            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render("human")


            if frame_idx % int(max_frames/10) == 0:
                print(
                    'frame : {}/{}, \t last rew : {}, \t rew loss : {}'.format(
                        frame_idx, max_frames, rewards[-1][1], model_optim.log['rew_loss'][-1]
                    )
                )

                pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break
        # for _ in range(100):
        #     if len(replay_buffer) > batch_size:
        #         sac.soft_q_update(batch_size)
        #         model_optim.update_model(batch_size, mini_iter=args.model_iter)
        if len(replay_buffer) > batch_size:
            print('ep rew', ep_num, episode_reward, model_optim.log['rew_loss'][-1], model_optim.log['loss'][-1])
            print('ssac loss', sac.log['value_loss'][-1], sac.log['policy_loss'][-1], sac.log['q_value_loss'][-1])
        rewards.append([frame_idx, episode_reward])
        ep_num += 1
    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
