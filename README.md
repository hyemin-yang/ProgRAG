# ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs
<img width="4335" height="2429" alt="model" src="https://github.com/user-attachments/assets/109c2dd1-9c31-46f5-a496-1b1eab7d3add" />

---
# How to run
## 1) Download Datasets and Checkpoints
To run experiments, download the required Knowledge Graph datasets and checkpoints for:
- Relation Retriever
- Triple Retriever : GNN, MPNet
  
Alternatively, you can preprocess the datasets using the following commands:
```
python3 graph_preprocess.py
python3 GNN/get_emb.py
```

📌 Note: Preprocessed datasets and checkpoints will be publicly available soon.

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
