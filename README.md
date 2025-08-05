# ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs
![model1](https://github.com/user-attachments/assets/dfcd36eb-fb31-4cf0-8d1c-5711777f1296)

---
# How to run
## 1) Installation
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

## 2) Download Datasets and Checkpoints
To run experiments, download the required Knowledge Graph datasets and checkpoints for:
- Relation Retriever
- Triple Retriever : GNN, MPNet
  
Alternatively, you can preprocess the datasets using the following commands:
```
python3 graph_preprocess.py
python3 GNN/get_emb.py
```
You can train the GNN and MPNet (Triple Retrievers) using the following commands:
```
python3 GNN/gnn_train.py
python3 MPNet/[dataset].sh 
```

📌 Note: Preprocessed datasets and checkpoints will be publicly available soon.


## 3) Run
```
python main.py --dataset [DATASET_NAME]
```
Replace [DATASET_NAME] with the name of the dataset you want to use (e.g., webqsp, cwq, etc.).
