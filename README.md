# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation



# Task Execution

## Overview  
This project demonstrates a sequence of Open-Vocabulary Mobile Manipulation (OVMM) that can be executed by a mobile manipulator robot.  

The example task sequence includes:  
1. Throw the can into the trash bin  
2. Move the pot near the red mug  
3. Put the mug on the shelf  
4. Open the cabinet  
5. Open the dishwasher  


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
    pip install -r requirements.txt
    pip install -e .
    ```

- Download the **[sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)** model and place it in the `affordance/` folder.

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
    python base_proposal/scripts/basic.py task=NM
    ```
## 3D Content Pack Download 
1. [OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K/tree/main/OmniGibson)
    -  Move the OmniGibson to the `../base_proposal/` folder.
  ```
├── OmniGibson/
└── base_proposal/
    ├── base_proposal/                
    └── README.md                      
  ```

2. [Base Materials Pack (Plane material)](https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html#basematerials)

  - Unzip them and put them into the Props folder.
  ```
└── base_proposal/base_proposal/base_proposal/usd/Props/
    ├── Base_Materials_NVD@10013/   
    ├── building/                      
    ├── Shapenet/                  
    └── YCB/                      
  ```

     

