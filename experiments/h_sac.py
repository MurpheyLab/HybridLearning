#! /usr/bin/env python
"""
imports
"""
# general
import numpy as np
import pickle
from datetime import datetime

import sys
import os
import signal
import traceback
sys.path.append('../')

import argparse
from copy import copy, deepcopy
import time

# model
import torch

from saclib import SoftActorCritic, PolicyNetwork, ReplayBuffer

from hltlib import StochPolicyWrapper, ModelOptimizer, Model, SARSAReplayBuffer

from mpclib import PathIntegral

# ros
import rospy
from sawyer_reacher_clutter import sawyer_env # reacher in clutter
# from sawyer_reacher_target import sawyer_env # reacher_target
# from sawyer_reacher import sawyer_env # reacher
# from sawyer_pusher import sawyer_env # pusher

"""
arg parse things
"""
parser = argparse.ArgumentParser()
# parser.add_argument('--env',        type=str,   help=envs.getlist())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=6000) # 6000 for last paper
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4) # was 3e-3 for last paper
parser.add_argument('--policy_lr',  type=float, default=3e-4) # was 3e-3 for last paper
parser.add_argument('--value_lr',   type=float, default=3e-4) # was 3e-4 for last paper
parser.add_argument('--soft_q_lr',  type=float, default=3e-4) # was 3e-4 for last paper

parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--model_iter', type=int, default=5) # 5 for sim # was 3 for last paper
parser.add_argument('--trajectory_samples', type=int, default=40) # 60 for sim # was 20 for last paper
parser.add_argument('--lam',  type=float, default=0.1)
parser.add_argument('--mode', type=str, default="hybrid")
parser.add_argument('--rate', type=int, default=5)
# parser.add_argument('--seed', type=int, default=5)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    try:
        rospy.init_node('h_sac')

        env = sawyer_env()

        env_name = 'clutter_200steps_10H'
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        if args.mode == "hybrid":
            path = './data/sawyer_' + env_name +  '/' + 'h_sac/' + date_str + '_' + env_name + "/"
        elif args.mode == "sac":
            path = './data/sawyer_' + env_name +  '/' + 'sac/' + date_str + '_' + env_name + "/" # policy only
        elif args.mode == "mpc":
            path = './data/sawyer_' + env_name +  '/' + 'mpc/' + date_str + '_' + env_name + "/" # model only
        else:
            raise ValueError('invalid mode entered')

        if os.path.exists(path) is False:
            os.makedirs(path)

        with open(path + 'args.txt','w') as f:
            f.write(str(args))
        f.closed

        action_dim = env.action_dim
        state_dim  = env.state_dim
        hidden_dim = 128
        replay_buffer_size = 10000


        # np.random.seed(args.seed)
        # # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if (args.mode == "mpc") or (args.mode == "hybrid"):
            model = Model(state_dim, action_dim, def_layers=[200])
            model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
            model_optim = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr, lam=0.)

        if (args.mode == "sac") or (args.mode == "hybrid"):
            replay_buffer = ReplayBuffer(replay_buffer_size)
            policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
            sac = SoftActorCritic(policy=policy_net,
                                  state_dim=state_dim,
                                  action_dim=action_dim,
                                  replay_buffer=replay_buffer,
                                  policy_lr=args.policy_lr,
                                  value_lr=args.value_lr,
                                  soft_q_lr=args.soft_q_lr)

        if args.mode == "hybrid":
            hybrid_policy = StochPolicyWrapper(model, policy_net,
                                                samples=args.trajectory_samples,
                                                t_H=args.horizon,
                                                lam=args.lam)
        if args.mode == "mpc":
            planner = PathIntegral(model,
                                    samples=args.trajectory_samples,
                                    t_H=args.horizon,
                                    lam=args.lam,
                                    eps=0.5)
        max_frames = args.max_frames
        max_steps  = args.max_steps
        frame_skip = args.frame_skip

        frame_idx  = 0
        rewards    = []
        success    = []
        batch_size = 128

        ep_num = 0
        fail = False

        rate=rospy.Rate(args.rate)

        while (frame_idx < max_frames) and not(fail):
            state = env.reset()
            if state is None:
                break
            if args.mode == "sac":
                action = policy_net.get_action(state.copy()) # policy only
            elif args.mode == "mpc":
                planner.reset()
                action = planner(state.copy())
            elif args.mode == "hybrid":
                hybrid_policy.reset()
                action = hybrid_policy(state.copy())

            episode_reward = 0
            episode_success = 0
            episode_stuck = 0

            for step in range(max_steps):

                if np.isnan(action).any():
                    print('got nan')
                    # print(replay_buffer.buffer)
                    fail = True
                    break
                    # env.reset(final=True)
                    # os._exit(0)
                else:
                    for _ in range(frame_skip):
                        rate.sleep()
                        next_state, reward, done, stuck, outside_boundary = env.step(action.copy()) # added clipped

                    if args.mode == "sac":
                        next_action = policy_net.get_action(next_state.copy())
                        replay_buffer.push(state, action, reward, next_state, done)
                        if len(replay_buffer) > batch_size:
                            sac.update(batch_size)
                    elif args.mode == "mpc":
                        next_action = planner(next_state.copy())
                        model_replay_buffer.push(state, action, reward, next_state, next_action, done)
                        if len(model_replay_buffer) > batch_size:
                            model_optim.update_model(batch_size, mini_iter=args.model_iter, verbose=False)
                    elif args.mode == "hybrid":
                        next_action = hybrid_policy(next_state.copy())
                        replay_buffer.push(state, action, reward, next_state, done)
                        model_replay_buffer.push(state, action, reward, next_state, next_action, done)
                        if len(replay_buffer) > batch_size:
                            # sac.soft_q_update(batch_size)
                            sac.update(batch_size)
                            model_optim.update_model(batch_size, mini_iter=args.model_iter, verbose=False)

                    # print(frame_idx,ep_num)
                    # print(next_state, state)
                    state = next_state.copy()
                    action = next_action.copy()
                    episode_reward += reward
                    frame_idx += 1

                    if frame_idx % (max_frames//10) == 0:
                        # last_reward = rewards[-1][1] if len(rewards)>0 else 0
                        # print(
                        #     'frame : {}/{}, \t last rew: {}'.format(
                        #         frame_idx, max_frames, last_reward
                        #     )
                        # )
                        print('saving model and reward log')
                        pickle.dump(success, open(path + 'success_data'+ '.pkl', 'wb'))
                        pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                        if args.mode != 'mpc':
                            torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                        if args.mode != 'sac':
                            torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')
                    if done:
                        episode_success = 1
                        print('done loop')
                        break
                    elif stuck:
                        print('stuck on edge of boundary')
                        episode_stuck = 1
                        break
                    elif outside_boundary:
                        print('pushed bock outside boundary')
                        break
                    # else:
                    #     rate.sleep()

            rewards.append([frame_idx, episode_reward])
            success.append([frame_idx, ep_num, episode_success, episode_stuck, outside_boundary])
            ep_num += 1

            last_reward = rewards[-1][1] if len(rewards)>0 else 0
            print(
                'frame : {}/{}, \t ep:{} \t last rew: {} '.format(
                    frame_idx, max_frames, ep_num, last_reward
                )
            )
        print('saving final data set')
        print(success)
        print(rewards)
        pickle.dump(success, open(path + 'success_data'+ '.pkl', 'wb'))
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        if (args.mode == "sac") or (args.mode == "hybrid"):
            torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
            pickle.dump(replay_buffer.buffer, open(path + 'replay_buffer'+ '.pkl', 'wb'))
        if (args.mode == "mpc") or (args.mode == "hybrid"):
            torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
            pickle.dump(model_replay_buffer.buffer, open(path + 'model_replay_buffer'+ '.pkl', 'wb'))

        env.reset(final=True)

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)
