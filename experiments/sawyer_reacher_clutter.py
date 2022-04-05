#! /usr/bin/env python
"""
imports
"""
# general
import numpy as np
import time
from copy import copy, deepcopy
import warnings

# ros
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from std_msgs.msg import Header, Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
from tf_conversions import transformations

# sawyer
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from intera_core_msgs.msg import JointCommand, EndpointState
from sawyer.msg import RelativeMove, Reward
from intera_interface import RobotParams, settings
import intera_dataflow

# pykdl
from sawyer_pykdl import sawyer_kinematics

class sawyer_env(object):
    def __init__(self):
        '''
        env params
        '''
        self.num_clutter = 5
        self.action_dim = 2
        self.state_dim = 2*(self.num_clutter+1)

        '''
        controller
        '''
        # set up ik solver
        self.iksvc = rospy.ServiceProxy("ExternalTools/right/PositionKinematicsNode/IKService", SolvePositionIK)
        rospy.wait_for_service("ExternalTools/right/PositionKinematicsNode/IKService", 5.0)

        # set up py_kdl
        self.py_kdl = sawyer_kinematics("right")

        # set up sawyer (in place of limb class)
        self._joint_names = RobotParams().get_joint_names("right")
        self._joint_angle = dict()
        self._joint_velocity = dict()
        # self._joint_effort = dict()
        self._tip_states = None

        # self._command_msg = JointCommand()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pub_joint_cmd = rospy.Publisher('/robot/limb/right/joint_command',JointCommand,tcp_nodelay=True,queue_size=1)
        self._pub_joint_cmd_timeout = rospy.Publisher('/robot/limb/right/joint_command_timeout',Float64,latch=True,queue_size=1)
        self._pub_speed_ratio = rospy.Publisher('/robot/limb/right/set_speed_ratio', Float64, latch=True, queue_size=1)
        _joint_state_sub = rospy.Subscriber('robot/joint_states',JointState,self._on_joint_states,queue_size=1,tcp_nodelay=True)
        _tip_states_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',EndpointState,self._on_tip_states,queue_size=1,tcp_nodelay=True)

        # set up controller
        # self.alpha = 0.2 # [0,1] # reacher is 0.8, first set of data was 0.2
        self.reset_joint_dict = dict()
        for name in self._joint_names:
            self.reset_joint_dict[name] = 0.0
        self.delta_theta = deepcopy(self.reset_joint_dict)

        # initalize home configuration
        home_orientation_quat = Quaternion(0, 1, 0, 0)
        home_orientation = (home_orientation_quat.x, home_orientation_quat.y,
                            home_orientation_quat.z, home_orientation_quat.w)
        self.home_orientation_rpy = np.asarray(transformations.euler_from_quaternion(home_orientation))

        # home_pose = [ -0.2437802734375, -0.4621513671875, -0.00378515625, 1.6477802734375, -0.120876953125, 0.4112978515625, 1.580935546875]
        home_pose = [-0.2694052734375, -0.5595732421875, -0.0193857421875, 1.973201171875, -0.256984375, 0.1513359375, 1.6200693359375]
        self.home_joints = dict(zip(self._joint_names, home_pose))

        # wait for first response to joint subscriber
        self.got_joints = False
        while (self.got_joints == False):
            time.sleep(0.2)
        self.desired_theta_dot = deepcopy(self.reset_joint_dict)

        self.raw_command = RelativeMove()
        # self.filtered_command = RelativeMove()
        self.reset_test = False
        print('controller setup complete')
        '''
        sawyer_env
        '''
        # set up ros
        self.move = rospy.Publisher('/test/relative_move',RelativeMove,queue_size=1)
        self.reward = rospy.Publisher('/test/reward',Reward,queue_size=1)
        self.listener = tf.TransformListener() # default = interpoation: True, cache_duration = 10.0
        rospy.Service('/test/done', Trigger, self.doneCallback)

        # set up tf
        self.got_pose = False
        while (self.got_pose == False):
            time.sleep(0.2)

        self.state = np.zeros(self.state_dim)
        self.clutter = np.zeros((2,self.num_clutter))
        self.block_start = np.zeros(2)
        self.block_pose = np.zeros(2)
        self.check_edges = np.zeros(3)
        # self.setup_transforms()
        self.get_transforms()
        clipped = self.update_velocities()
        print('sawyer env setup complete')

    '''
    from limb class
    '''
    def _on_joint_states(self, msg):
        self.got_joints = True
        for idx, name in enumerate(msg.name):
            if name in self._joint_names:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                # self._joint_effort[name] = msg.effort[idx]

    def joint_angles(self):
        return deepcopy(self._joint_angle)

    def _on_tip_states(self, msg):
        self.got_pose = True
        self._tip_states = deepcopy(msg)

    def set_joint_positions(self, positions):
        _command_msg = JointCommand()
        _command_msg.names = positions.keys()
        _command_msg.position = positions.values()
        _command_msg.mode = JointCommand.POSITION_MODE
        _command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(_command_msg)

    def move_to_joint_positions(self, positions, timeout=15.0,
                                threshold=settings.JOINT_ANGLE_TOLERANCE,
                                test=None):
        cmd = self.joint_angles()

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._joint_angle[joint])
            return joint_diff

        diffs = [genf(j, a) for j, a in positions.items() if
                 j in self._joint_angle]
        fail_msg = "limb failed to reach commanded joint positions."
        def test_collision():
            # if self.has_collided():
            #     rospy.logerr(' '.join(["Collision detected.", fail_msg]))
            #     return True
            return False
        self.set_joint_positions(positions)
        intera_dataflow.wait_for(
            test=lambda: test_collision() or \
                         (callable(test) and test() == True) or \
                         (all(diff() < threshold for diff in diffs)),
            timeout=timeout,
            timeout_msg=fail_msg,
            rate=100,
            raise_on_error=False,
            body=lambda: self.set_joint_positions(positions)
            )

    def set_joint_velocities(self, velocities):
        _command_msg = JointCommand()
        _command_msg.names = velocities.keys()
        _command_msg.velocity = velocities.values()
        _command_msg.mode = JointCommand.VELOCITY_MODE
        _command_msg.header.stamp = rospy.Time.now()
        self._pub_joint_cmd.publish(_command_msg)

    # def set_joint_torques(self, torques):
    #     _command_msg = JointCommand()
    #     _command_msg.names = torques.keys()
    #     _command_msg.effort = torques.values()
    #     _command_msg.mode = JointCommand.TORQUE_MODE
    #     _command_msg.header.stamp = rospy.Time.now()
    #     self._pub_joint_cmd.publish(_command_msg)

    '''
    controller
    '''
    def ee_vel_to_joint_vel(self,_data,orientation):
        # calculate jacobian_pseudo_inverse
        data,clipped = self.clip_velocities(_data)

        jacobian_ps = self.py_kdl.jacobian_pseudo_inverse(joint_values=None)

        xdot = np.zeros(6)
        xdot[0] = data.dx
        xdot[1] = data.dy
        xdot[2] = data.dz
        # xdot[3] = orientation[0]
        # xdot[4] = orientation[1]
        # xdot[5] = orientation[2]

        desired_theta_dot = np.matmul(jacobian_ps,xdot)

        for i in range(len(self._joint_names)):
            self.desired_theta_dot[self._joint_names[i]] = desired_theta_dot[0,i]
        return clipped

    def check_orientation(self):
        current_orientation = self._tip_states.pose.orientation
        quaternion = (current_orientation.x, current_orientation.y,
                      current_orientation.z, current_orientation.w)
        current_orientation_rpy = np.asarray(transformations.euler_from_quaternion(quaternion))
        correction = current_orientation_rpy-self.home_orientation_rpy
        correction[2] *= -1
        for i in range(3):
            if correction[i] < -np.pi:
                correction[i] += 2*np.pi
            elif correction[i] > np.pi:
                correction[i] -= 2*np.pi
        return correction

    def clip_velocities(self,action):
        # max_x = 0.75
        # min_x = 0.5 # 0.45
        # max_y = 0.25 #0.3
        # min_y = -0.2 #-0.25
        max_x = 0.7
        min_x = 0.45
        max_y = 0.15 #0.3
        min_y = -0.1 #-0.25
        slow = 1.0
        current_pose = deepcopy(self._tip_states)

        clipped = 0.
        if (current_pose.pose.position.x > max_x):
            clipped += 1
            action.dx = np.clip(action.dx, -1.*slow,0.)
            if self.state[-2] > 0.2: # block out of reach in +x direction
                self.check_edges[2] += 1
        elif (current_pose.pose.position.x < min_x):
            clipped += 10
            action.dx = np.clip(action.dx, 0.,1.*slow)
            if self.state[-2] < 0.2: # block out of reach in -x direction
                self.check_edges[2] += 1
        if (current_pose.pose.position.y > max_y):
            clipped += 100
            action.dy = np.clip(action.dy, -1.*slow,0.)
            if self.state[-1] > 0.2: # block out of reach in +y direction
                self.check_edges[2] += 1
        elif(current_pose.pose.position.y < min_y):
            clipped += 1000
            action.dy = np.clip(action.dy, 0.,1.*slow)
            if self.state[-1] < 0.2: # block out of reach in -y direction
                self.check_edges[2] += 1

        if (current_pose.pose.position.z > -0.05):
            action.dz = - 0.01
        elif (current_pose.pose.position.z < -0.07):
            action.dz = 0.01

        if clipped > 0:
            if self.check_edges[0] == clipped:
                self.check_edges[1] += 1
            else:
                self.check_edges[0] = clipped
                self.check_edges[1] = 1
        else:
            self.check_edges = np.zeros(3)
        return action, clipped

    def move_vertical(self,direction):
        # solve inverse kinematics
        ikreq = SolvePositionIKRequest()

        current_pose = self._tip_states.pose # get current state

        pose = current_pose
        if direction == 1:
            pose.position.z -= 0.06
        else:
            pose.position.z += 0.06

        # create stamped pose with updated pose
        poseStamped = PoseStamped()
        poseStamped.header = Header(stamp=rospy.Time.now(), frame_id='base')
        poseStamped.pose = pose

        # Add desired pose for inverse kinematics
        ikreq.pose_stamp.append(poseStamped)
        ikreq.tip_names.append('right_hand') # for each pose in IK

        limb_joints = copy(self.home_joints)
        try:
            resp = self.iksvc(ikreq)

            # Check if result valid, and type of seed ultimately used to get solution
            if (resp.result_type[0] > 0):
                limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
                # rospy.loginfo("Response Message:\n%s", resp)
                self.desired_theta = limb_joints
            else:
                rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
                rospy.logerr("Result Error %d", resp.result_type[0])

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))

        return limb_joints

    def update_velocities(self):
        if self.reset_test == True:
            # move up
            vertical_joints = self.move_vertical(0)
            self._pub_speed_ratio.publish(Float64(0.1))
            self.move_to_joint_positions(vertical_joints)
            time.sleep(1)

            # move to home
            self._pub_speed_ratio.publish(Float64(0.2))
            self.move_to_joint_positions(copy(self.home_joints))
            time.sleep(1)

            # move down
            vertical_joints = self.move_vertical(1)
            self._pub_speed_ratio.publish(Float64(0.1))
            self.move_to_joint_positions(vertical_joints)
            time.sleep(1)

            # reset parameters
            # self.delta_theta = deepcopy(self.reset_joint_dict)
            self.desired_theta_dot = deepcopy(self.reset_joint_dict)  # copy(self.home_joints)
            self.raw_command = RelativeMove()
            # self.filtered_command = RelativeMove()
            self.reset_test = False
            print("Reset Pose")

        raw_orientation_correction = self.check_orientation()
        # self.filtered_command.dx = self.alpha*self.filtered_command.dx+(1-self.alpha)*self.raw_command.dx
        # self.filtered_command.dy = self.alpha*self.filtered_command.dy+(1-self.alpha)*self.raw_command.dy
        # self.filtered_command.dz = self.alpha*self.filtered_command.dz+(1-self.alpha)*self.raw_command.dz
        clipped = self.ee_vel_to_joint_vel(self.raw_command,raw_orientation_correction*.25)
        self.set_joint_velocities(self.desired_theta_dot)
        self.move.publish(self.raw_command)
        return clipped

    '''
    sawery_env
    '''

    # def setup_transforms(self):
    #     x_idx = 0
    #     y_idx = 1
    #     lookups = ['clutter1','clutter2','clutter3','clutter4','clutter5','target','top'] # removed target
    #     all_coords = []
    #     try:
    #         for count,name in enumerate(lookups):
    #             all_coords.append(current_transform[x_idx])
    #             current_transform,_ = self.listener.lookupTransform( 'ee', name, rospy.Time(0))
    #             all_coords.append(current_transform[y_idx])
    #             if count < self.num_clutter:
    #                 self.clutter[0,count] = current_transform[x_idx]*10.
    #                 self.clutter[1,count] = current_transform[y_idx]*10.
    #         self.state = np.array(all_coords)*10.
    #     except:
    #         print("Check that all april tags are visible, didn't see state_dim ",name)
    #     # new
    #     try:
    #         current_transform,_ = self.listener.lookupTransform( 'target', 'top', rospy.Time(0))
    #         self.block_pose[0] = current_transform[x_idx]*10.
    #         self.block_pose[1] = current_transform[y_idx]*10.
    #     except:
    #         print("check target is visible")
    #     return self.state

    def get_transforms(self):
        x_idx = 0
        y_idx = 1
        clutter_lookups = ['clutter1','clutter2','clutter3','clutter4','clutter5']
        for count, name in enumerate(clutter_lookups):
            try:
                # lookupTransform(target_frame, source_frame, time) -> (position, quaternion)
                current_transform, _ = self.listener.lookupTransform( 'ee',name, rospy.Time(0))
                test = current_transform[x_idx] # try to see if you can access item before saving it to state
                self.clutter[0,count] = current_transform[x_idx]*10.
                self.clutter[1,count] = current_transform[y_idx]*10.
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Check that all april tags are visible, didn't see tag for:",name)
                pass
        new_clutter = self.clutter.copy() # copy so main clutter keeps order
        new_order = np.argsort(np.sum(np.square(new_clutter),axis=0)) # sort by distance from arm
        new_clutter = new_clutter[:,new_order] # reorder
        new_clutter = new_clutter.flatten(order='F') # flatten column-major
        self.state[:self.num_clutter*2] = new_clutter.copy()

        # name = 'target'
        # state_idx = 10
        # try:
        #     current_transform, _ = self.listener.lookupTransform( 'ee', name rospy.Time(0))
        #     test = current_transform[x_idx] # try to see if you can access item before saving it to state
        #     self.state[state_idx] = current_transform[x_idx]*10.
        #     self.state[state_idx+1] = current_transform[y_idx]*10.
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     print("Check that all april tags are visible, didn't see tag for ",name)
        #     pass

        name = 'top'
        state_idx = 10
        try:
            current_transform, _ = self.listener.lookupTransform('ee', name,  rospy.Time(0))
            test = current_transform[x_idx] # try to see if you can access item before saving it to state
            self.state[state_idx] = current_transform[x_idx]*10.
            self.state[state_idx+1] = current_transform[y_idx]*10.
            current_transform,_ = self.listener.lookupTransform('target', name,  rospy.Time(0))
            test = current_transform[x_idx] # try to see if you can access item before saving it
            self.block_start = self.block_pose.copy() # save last before updating
            self.block_pose[0] = current_transform[x_idx]*10.
            self.block_pose[1] = current_transform[y_idx]*10.
            return name
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Check that all april tags are visible, didn't see tag for ",name)
            pass


        # block_lookups = ['top', 'block1','block2','block3','block4'] # removed target
        # state_idx = 10 # removed target
        # for count, name in enumerate(block_lookups):
        #     try:
        #         current_transform, _ = self.listener.lookupTransform( 'ee',name, rospy.Time(0))
        #         test = current_transform[x_idx] # try to see if you can access item before saving it to state
        #         self.state[state_idx] = current_transform[x_idx]*10.
        #         self.state[state_idx+1] = current_transform[y_idx]*10.
        #         # if count > 0: # subtract 1/2 cube depth from x if looking at a tag on the side of the block
        #         #     self.state[state_idx] = (current_transform[x_idx]-0.024)*10.
        #         current_transform,_ = self.listener.lookupTransform( 'target', name, rospy.Time(0))
        #         test = current_transform[x_idx] # try to see if you can access item before saving it
        #         self.block_start = self.block_pose.copy() # save last before updating
        #         self.block_pose[0] = current_transform[x_idx]*10.
        #         self.block_pose[1] = current_transform[y_idx]*10.
        #         return name
        #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #         pass

        return None # no transforms


    def reward_function(self,_a):
        # dx +/- 0.02

        current_state = self.state.copy()
        dx_ArmToBlock = current_state[-2] # second to last entry
        dy_ArmToBlock = current_state[-1] # last entry

        arm_to_block = np.sqrt(dx_ArmToBlock**2+dy_ArmToBlock**2)

        block_move_dist = np.sqrt(np.sum(np.square(self.block_start-self.block_pose)))

        reward = 0
        done = False
        thresh = 0.2

        reward -= arm_to_block
        reward -= block_move_dist

        if (abs(dy_ArmToBlock) < 0.2) and (arm_to_block < thresh):
            done = True
            # reward += 10
            print('Reached goal!')

        vel_cost = np.sum(_a**2)*0.01
        reward -= np.clip(vel_cost,-1.,1.)

        reward_msg = Reward()
        reward_msg.reward = reward
        reward_msg.distance1 = arm_to_block
        reward_msg.distance2 = block_move_dist
        self.reward.publish(reward_msg)

        return reward, done, reward_msg

    def reset(self,final=False):
        self.reset_test = True
        self.update_velocities()
        if not final:
            o = raw_input("Press enter to continue, press any letter then enter to quit\t")
            if o == '':
                pass
            else:
                return None
            # self.setup_transforms()
            self.check_edges = np.zeros(3) # reset edge checker
            self.get_transforms()
            return self.state.copy()

    def step(self, _a):
        if (self.reset_test == False):
            # theta = (np.pi/4)*np.clip(_a[2],-1,1)  # keep april tags in view
            action = 0.1*np.clip(_a, -1., 1.)
            # publish action input
            pose = RelativeMove()
            pose.dx = action[0]
            pose.dy = action[1]
            # pose.dtheta = theta

            self.raw_command = pose
            # print('step function: raw ', _a[:2], 'clipped ',action[:2])

            # get new state
            side = self.get_transforms()
            reward, done, reward_msg = self.reward_function(_a)
        else:
            done = True
            reward, _ , reward_msg = self.reward_function(_a)

        # next_reward = Reward()
        # next_reward.reward = reward
        # self.reward.publish(next_reward)

        clipped = self.update_velocities()
        # if (clipped>0):
        #     reward -= clipped

        # reward_msg.reward = reward
        # self.reward.publish(reward_msg)

        return self.state.copy(), reward, done, False, (self.check_edges[2] > 3) #(clipped>0) (self.check_edges[1] > 10)

    def doneCallback(self,req):
        self.reset_test = True
        print('manual done called')
        return TriggerResponse(success=True,message="Done callback complete")
