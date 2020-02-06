#! /usr/bin/env python
"""
Set up sawyer environment

Used by h_sac.py
"""
# general
import numpy as np
import time
from copy import copy, deepcopy

# ros
import rospy
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Trigger, TriggerResponse

# sawyer
from sawyer.msg import RelativeMove, Reward
from intera_core_msgs.msg import EndpointState,EndpointStates
from intera_interface import Gripper

class sawyer_env(object):
    def __init__(self):
        # set up ros
        self.move = rospy.Publisher('/puck/relative_move',RelativeMove,queue_size=1)
        self.reward = rospy.Publisher('/puck/reward',Reward,queue_size=1)
        self.reset_arm = rospy.ServiceProxy('/puck/reset', Trigger)
        rospy.wait_for_service('/puck/reset', 5.0)
        rospy.Service('/puck/done', Trigger, self.doneCallback)

        # set up flags
        self.reset_test = False

        # set up sawyer
        self.tip_name = "right_hand"
        self._tip_states = None
        _tip_states_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',EndpointState,self._on_tip_states,queue_size=1,tcp_nodelay=True)
        limb = "right"
        self.gripper = Gripper(limb + '_gripper')

        # set up tf
        self.got_pose = False
        while (self.got_pose == False):
            # print('waiting')
            time.sleep(0.2)

        self.state = np.zeros(9)
        self.update_state()

    def _on_tip_states(self, msg):
        self.got_pose = True
        self._tip_states = deepcopy(msg)

    def update_state(self):
        target = np.zeros(9)
        target[0:3] = np.array([0.618718108914, 0.0361612427719, 0.143426261079])

        c_tip_state = copy(self._tip_states)

        ee = np.array([c_tip_state.pose.position.x, c_tip_state.pose.position.y, c_tip_state.pose.position.z,
                       c_tip_state.wrench.force.x,  c_tip_state.wrench.force.y,  c_tip_state.wrench.force.z,
                       c_tip_state.wrench.torque.x, c_tip_state.wrench.torque.y, c_tip_state.wrench.torque.z])

        self.state = ee-target

    def reset(self):
        self.reset_test = False
        resp = self.reset_arm()
        o = raw_input("Enter '0' to open gripper, otherwise press enter to continue ")
        if (o == '0'):
            self.gripper.open()
            raw_input("Press enter to close gripper")
            self.gripper.close("Press enter to continue")
            raw_input()
        self.update_state()
        return self.state.copy()

    def step(self, _a):
        if (self.reset_test == False):
            # theta = (np.pi/4)*np.clip(_a[3],-1,1)  # keep april tags in view
            action = 0.2*np.clip(_a, -1,1)
            # publish action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            pose.dz = action[2]
            # pose.dtheta = theta
            self.move.publish(pose)

            # get new state
            self.update_state()
            reward, done = self.reward_function()
        else:
            done = True
            reward, _ = self.reward_function()

        next_reward = Reward()
        next_reward.reward = reward
        next_reward.distance = reward
        self.reward.publish(next_reward)

        return self.state.copy(), reward, done

    def reward_function(self):
        [distance_dx, distance_dy, distance_dz,
         force_dx, force_dy, force_dz,
         torque_dx, torque_dy, torque_dz] = self.state.copy()

        l2norm = distance_dx**2+distance_dy**2+distance_dz**2
        distance = np.sqrt(distance_dx**2+distance_dy**2+distance_dz**2+1e-3)
        force = np.sqrt(force_dx**2+force_dy**2+force_dz**2)
        torque = np.sqrt(torque_dx**2+torque_dy**2+torque_dz**2)

        reward = 0
        done = False
        thresh = 0.032

        # reward = -distance-l2norm
        reward = -distance
        # reward += -torque*1e-6
        # reward += -force*1e-6

        if (distance < thresh):
            done = True
            # reward += 10
            print('Reached goal!')


        # rospy.loginfo("action reward: %f", reward)
        # rospy.loginfo("distance: %f", distance)
        # rospy.loginfo("torque: %f", torque)
        # rospy.loginfo("force: %f", force)

        return reward, done

    def doneCallback(self,req):
        self.reset_test = True
        print('manual done called')
        return TriggerResponse(success=True, message="Done callback complete")
