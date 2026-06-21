
# SurgVLA-Bench: A Vision-Language-Action Model Evaluation Benchmark for Laparoscopic Surgical Robotics

This repository contains the official code and dataset links for the paper **"SurgVLA-Bench: Towards Evaluating Vision-Language-Action Models for Laparoscopic Surgical Robotics"**.

SurgVLA-Bench is a comprehensive benchmark specifically designed to evaluate Vision-Language-Action (VLA) models in laparoscopic surgical environments. Based on the SurRoL simulation platform, we constructed a hierarchical task taxonomy ranging from atomic actions to complete surgical procedures.

## 📢 News

* **[2026-06]** Our paper has been accepted to **MICCAI 2026**!
* **[2026-06]** Official release of the codebase and the SurgVLA dataset.

## 📊 Dataset

Unlike general robotics, the surgical domain has previously lacked standard datasets suitable for VLA model training and evaluation. To address this, we provide a comprehensive standardized dataset supporting multiple mainstream formats, including RLDS and LeRobot.

The dataset contains over 800 complete trajectories across 8 surgical tasks, totaling approximately 40,000 action frames.

You can access and download the full dataset from the `main` branch of our repository:
🔗 **[Kanden1112/surg-vla-dataset](https://huggingface.co/datasets/Kanden1112/surg-vla-dataset/tree/main)**

## 🛠️ Installation & Environment Setup

To systematically evaluate different VLA paradigms, we currently benchmarked autoregressive models (OpenVLA) and flow matching models ($\pi_0$, $\pi_{0.5}$, and SmolVLA). We plan to add benchmarks for more models in the future. Because these architectures have conflicting dependencies, **we provide three separate Conda environments** for testing the different models.

Our project is based on SurRoL, but we have integrated the core codebase of several VLA models directly into our SurgVLA repository. Therefore, for the models included in our paper, you only need to install their corresponding script dependencies within our project's base environment.

Taking OpenVLA as an example. First, clone this repository to your local machine:

```bash
conda create -n Surg_Open python=3.10 -y # The pi series requires python=3.11
conda activate Surg_Open
git clone https://github.com/exploding4994/SurgVLA
cd SurgVLA
pip install -e .
cd surrol
pip install -r Open_requirements.txt

```

*(The environment configuration for other models follows a similar process; simply use the corresponding `requirements` configuration file.)*

## 🏥 Task Taxonomy

Our benchmark features 8 distinct tasks, categorized into three levels based on clinical relevance:

* **Level 1: Atomic Tasks**
* `T1-1 Gauze Pick`: Pick up the gauze.
* `T1-2 Needle Pick`: Pick up the needle.
* `T1-3 Electrocoagulation`: Touch the red blood spot on the kidney.


* **Level 2: Conditional Tasks**
* `T2-1 Gauze Pick`: Pick up the gauze (with distractors).
* `T2-2 Needle Pick`: Pick up the needle (with distractors).
* `T2-3 Vessel Clipping`: Clip the red blood point on the vein.


* **Level 3: Composite Tasks**
* `T3-1 Pick and Place`: Pick up the gauze and place it on the block.
* `T3-2 Hemostasis`: Pick up the gauze and place it on the red target point on the spleen.



## 🚀 Training

To ensure fairness in evaluation and consistency with the model ecosystems, SurgVLA-Bench strictly follows the training methods and architectural standards recommended by the original projects of each VLA model during fine-tuning for all tasks. For our tests across all models, we use versions that have converged after fine-tuning with the LoRA method.

For a deeper understanding or to customize the underlying training logic of each model, please refer to their respective official GitHub repositories:

* **OpenVLA**: [https://github.com/openvla/openvla](https://github.com/openvla/openvla)
* **$\pi_0$ / $\pi_{0.5}$ (Physical Intelligence)**: [https://github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
* **SmolVLA (Hugging Face LeRobot)**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

## 🧪 Evaluation & Testing

The testing and inference process has been fully integrated into this project. Below is an example using the evaluation of **OpenVLA** on **Task 1 (`T1-1 Gauze Pick`)**.

Please ensure you have activated the corresponding Conda environment, adjust the script paths as needed, and run the following command to start the evaluation:

```bash
cd surrol/tasks
python task1_test_open_eval.py

```

## 🤝 Acknowledgements

This project is built upon the [SurRoL](https://github.com/med-air/SurRoL) simulation platform. We sincerely thank the developers for their open-source contributions to surgical robot learning.
