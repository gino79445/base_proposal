# Nav2Manip


## Installation



- Install isaac-sim on your PC by following the procedure outlined here: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html\
**Note:** This code was tested on isaac-sim **version 2022.1.0**
- Follow the isaac-sim python conda environment installation at: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda\
Note that we use a modified version of the isaac-sim conda environment `isaac-sim-lrp` which needs to be used instead and is available at `learned_robot_placement/environment.yml`. Don't forget to source the `setup_conda_env.sh` script in the isaac-sim directory before running experiments. (You could also add it to the .bashrc)
- The code uses pinocchio [3] for inverse kinematics. The installation of pinoccio is known to be troublesome but the easiest way is to run `conda install pinocchio -c conda-forge` after activating the `Nav2Manip ` conda environment.

### Setup RL algorithm and environments
- Install this repository's python package:
    ```
    cd learned_robot_placement
    pip install -e .
    ```

## Experiments

### Launching the experiments
- Activate the conda environment:
    ```
    conda activate Nav2Manip
    ```
- source the isaac-sim conda_setup file:
    ```
    source <PATH_TO_ISAAC_SIM>/isaac_sim-2022.1.0/setup_conda_env.sh
    ```
- To test the installation, an example random policy can be run:
    ```
    python learned_robot_placement/scripts/basic.py
    ```
