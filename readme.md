## Installation
- Simulations were run on python 3.6 (and 3.7) with pytorch 1.7.1 (both with and without GPU)  
- Other Python Simulation Dependencies: numpy, gym, pyyaml, pybullet, mujoco-py, roboschool
- Python Visualization Dependencies: matplotlib, ipython, jupyterlab, seaborns

## Environments Modifications
- Some versions of the `roboschool` environments do not properly initialize the swingup environment. If you run the  `RoboschoolInvertedPendulumSwingup` environment, and it appears to be doing the balance task, check `roboschool/gym_pendulums.py`
the `__init__()` function  for the `RoboschoolInvertedPendulumSwingup` task. The `__init__()` function should include `RoboschoolInvertedPendulum.__init__(self,swingup=True)`
- The `gym` Acrobot environment was modified to test in the continuous space and the reward function was updated to include a dependency on the action and the state. The details can be viewed in `/envs/continuous_acrobot.py`

## Copyright and License

The implementations of Hybrid Learning contained herein are copyright (C) 2021 - 2022 by Allison Pinosky and Todd Murphey, and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

Contact: apinosky@u.northwestern.edu

Lab Info:
Todd D. Murphey
https://murpheylab.github.io/
Northwestern University
