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
## 3D Content Pack Download
Download the following 3d model packs
1. [Commercial 3D Models Pack](https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html#commercial3dmodels)
2. [Residential 3D Models Pack](https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html#residential3dmodels)
3. [Base Materials Pack](https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html#basematerials)

- Unzip them and put them into the Props folder.
```
└── base_proposal/base_proposal/base_proposal/usd/Props/
    ├── Base_Materials_NVD@10013/   
    ├── building/                   
    ├── Commercial_NVD@10013/       
    ├── Residential_NVD@10012/      
    ├── Shapenet/                  
    └── YCB/                        
```
# build the enviroments
- Open the GUI:
    ```
    source <PATH_TO_ISAAC_SIM>/isaac_sim.sh
    ```
- Import 3d model (*.usd)
    - You can pull the *.usd from the following folders ( Commercial_NVD@10013 , Residential_NVD@10012 ...) to build the rooms.
    ```
    └── base_proposal/base_proposal/base_proposal/usd/Props/
        ├── Base_Materials_NVD@10013/   
        ├── building/                   
        ├── Commercial_NVD@10013/       
        ├── Residential_NVD@10012/      
        ├── Shapenet/                  
        └── YCB/                        
    ```
    
- You can build by referring to this *living_room* example.

  Example :``` base_proposal/base_proposal/base_proposal/usd/Props/Shapenet/living_room/models/model_normalized.usd```

- Save the built room in the Shapenet folder using the following format.
   ```
    └── base_proposal/base_proposal/base_proposal/usd/Props/Shapenet/ 
        └── <your_room_name>/   
            └── models/                   
                └── model_normalized.usd                      
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
    python base_proposal/scripts/basic.py task=NM
    ```
