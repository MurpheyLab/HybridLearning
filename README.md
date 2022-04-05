# Sawyer experiment for finding a block in clutter (of other blocks)

All experiments run with ROS Melodic and Python 2 on a Sawyer Robot from Rethink Robotics.

AprilTags for blocks and end-effector are specified in `/sawyer/block_tags1.yaml`

Sawyer setup, apriltag tracking, and visualizations need to be run with `roslaunch sawyer init_cam_n_track.launch`

To run the experiments, run `python /experiments/h_sac.py --mode 'hybrid'` Alternate modes are 'sac' and 'mpc' for the model-free and model-based methods respectively.

Finally, the data can be visualized with the notebook in `/experiments/data/make_figure.ipynb`
