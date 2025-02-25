# Nav2Manip


## Installation



- Download the isaac-sim **[version 4.2.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html)**.
  The downloaded folder contains the **setup_conda_env.sh** file.

     **Note:** This code was tested on isaac-sim **version 4.2.0** and python **version 3.10**

### setup the enviroment
- Install this repository's python package:
    ```
    conda create --name Nav2Manip python=3.10
    conda activate Nav2Manip
    conda install pinocchio -c conda-forge
    cd base_proposal/base_proposal/
    pip install -e .
    pip install open3d
    pip install openai
    pip install dotenv
    ```

## Experiments

### Launching the experiments
- Activate the conda environment:
    ```
    conda activate Nav2Manip
    ```
- source the isaac-sim conda_setup file:
    ```
    source <PATH_TO_ISAAC_SIM>/setup_conda_env.sh
    ```
- To test the installation, an example can be run:
    ```
    cd base_proposal/base_proposal/
    python  base_proposal/scripts/basic.py task=NM
    ```
