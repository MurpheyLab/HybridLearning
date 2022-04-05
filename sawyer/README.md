## General
# setup (one time)
copy sawyer/config/head_camera.yaml to ~/.ros/camera_info/.

# source robot
source ~/murphey_lab/sawyer_ws/src/SMPO/sawyer/robot.bash
# source rethink workspace (or wherever you have sawyer tools installed)
source ~/rethink_ws/devel/setup.bash
# check connetion
ping 10.42.0.2
# check status/enable/disable robot
rosrun intera_interface enable_robot.py -s

# source workspace
source ~/murphey_lab/sawyer_ws/devel/setup.bash

# run camera interface and run ik_solver
roslaunch sawyer init_cam_n_track.launch


# to control arm...
# 1. run test script
rosrun sawyer test_traj
# 2. reset to home pose
rosservice call /puck/reset
# 3. manually test
rostopic pub /puck/relative_move sawyer/RelativeMove "dx: 0.0
dy: 0.1"


# modified h_sac.py to run test

## old
# go to start position
# higher
rosrun intera_examples go_to_joint_angles.py -q -0.2 -0.6 0.0 1.9 0.0 0.3 1.571 -s 0.6
# lower
rosrun intera_examples go_to_joint_angles.py -q -0.1 -0.3 0.0 1.6 0.0 0.3 1.571 -s 0.8

# run joint action trajectory server
rosrun intera_interface joint_trajectory_action_server.py
# to check current arm position
rostopic echo /robot/limb/right/endpoint_state
