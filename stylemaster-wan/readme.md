# StyleMaster-Wan

This repository contains an implementation of **StyleMaster** T2V model upon the **Wan-2.1** base model. 


## 🚀 Getting Started

Follow these steps to set up the environment, install dependencies, and download the necessary models.

### 1. Open the Workspace

```bash
cd stylemaster-wan
```

### 2. Create and Activate Conda Environment

```bash
curl --proto '=https' --tlsv1.2 -sSf [https://sh.rustup.rs](https://sh.rustup.rs/) | sh
. "$HOME/.cargo/env"
conda create --name stylemaster python=3.10
conda activate stylemaster
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Download Checkpoints

Run the provided script to download the pre-trained checkpoints. 
```bash
python download_ckpt.py
```
After this step, you should have `checkpoints/` and `Models/`directory containing the necessary model files.

## 🎨 Inference

Once the setup is complete, you can easily generate images using the `inference_stylemaster.py` script.

```bash
python inference_stylemaster.py 
```
```bash
python inference_stylemaster_v2v.py 
```
## 🎓 Training

You can train your own StyleMaster adapter on a custom dataset to capture a specific style. The complete workflow for data processing and training is detailed in `script.sh`.

### General Workflow

The training process generally involves two main steps:

1.  **Data Preparation:** Refer to the `./data` folder for examples. 

2.  **Training Execution:** Once the dataset is prepared, you can launch the training script. 

## Acknowledgements

This project is an implementation of the **StyleMaster** method on the **Wan-2.1** model.

Our code heavily relies on [Diffsynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster), thanks for their contribution!

