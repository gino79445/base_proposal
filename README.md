# Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation

## 🔧 Project Overview

This project showcases a pipeline for **Open-Vocabulary Mobile Manipulation (OVMM)** with a mobile manipulator robot. It explores base placement strategies guided by **affordance maps** in a **coarse-to-fine** manner.

### OVMM tasks
The system supports executing complex, multi-step manipulation tasks in open environments:
1. Throw the can into the trash bin  
2. Move the pot near the red mug  
3. Put the mug on the shelf  
4. Open the cabinet  
5. Open the dishwasher  

---

## 🛠️ Installation Guide

### ⚙️ Requirements
- **Isaac Sim:** Version [4.2.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html)
  - The downloaded Isaac Sim folder contains the `setup_conda_env.sh` file, which is required to initialize its environment.   
- **Python:** 3.10

### ⚙️ Environment Setup
```bash
conda create --name Nav2Manip python=3.10
conda activate Nav2Manip
conda install pinocchio -c conda-forge

# Clone and install the repository
cd base_proposal/base_proposal/
pip install -r requirements.txt
pip install -e .
```

### ⚙️ Download Required Models
- Download the [sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it in:
  ```
  base_proposal/base_proposal/affordance/sam_vit_h.pth
  ```

---

## 💡 Running Experiments

1. Activate your environment:
   ```bash
   conda activate Nav2Manip
   ```

2. Source Isaac-Sim environment:
   ```bash
   source <PATH_TO_ISAAC_SIM>/setup_conda_env.sh
   ```

3. Run the example task:
   ```bash
   cd base_proposal/base_proposal/
   python base_proposal/scripts/basic.py task=NM
   ```

---

## 📦 3D Content Download

### 1. OmniGibson Assets
- Clone/download from: [OmniGibson GitHub](https://github.com/StanfordVL/BEHAVIOR-1K/tree/main/OmniGibson)
- Place it next to the base project directory:
  ```
  ├── OmniGibson/
  └── base_proposal/
      └── base_proposal/
  ```

### 2. Base Materials (Plane, Props, etc.)
- Download from [Omniverse Content Installation](https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html#basematerials)
- Unzip into:
  ```
  base_proposal/base_proposal/base_proposal/usd/Props/
      ├── Base_Materials_NVD@10013/
      ├── building/
      ├── Shapenet/
      └── YCB/
  ```

### 📁 Alternatively:
Download both `OmniGibson` and `Props` folders from this [Google Drive link](https://drive.google.com/drive/folders/1fCjWBXw-kdtnIv1Sl1UHT88W0paeg84t?usp=sharing)
