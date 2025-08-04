# ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs
<img width="3345" height="1387" alt="image" src="https://github.com/user-attachments/assets/df44c9a3-8bac-4f0d-82c5-13fda1f4b6c9" />

---
# How to run
## 1) Download Datasets and Checkpoints
To run experiments, download the required Knowledge Graph datasets and checkpoints for:
- Relation Retriever
- Triple Retriever : GNN, MPNet

📌 Note: Datasets and checkpoints will be publicly available soon.

## 2) Installation
```
# Create and activate a new environment
conda create -n ProgRAG python=3.8 -y
conda activate ProgRAG

# Install PyTorch with CUDA 11.8
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyG (PyTorch Geometric)
conda install pyg -c pyg

# Install other dependencies
conda install ninja easydict pyyaml -c conda-forge
pip install transformers sentence-transformers
```
## 3) Run
```
python main.py --dataset [DATASET_NAME]
```
Replace [DATASET_NAME] with the name of the dataset you want to use (e.g., webqsp, cwq, etc.).
