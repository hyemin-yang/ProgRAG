# [AAAI 2026] ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs
Codes for the paper titled ["ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs"](https://arxiv.org/pdf/2511.10240), published in the 40th Annual AAAI Conference on Artificial Intelligence (AAAI'2026).

---

![model](source/model.png)

---
# How to run
## 1) Installation
```
# 1. Create and activate a new conda environment
conda create -n ProgRAG python=3.8 -y
conda activate ProgRAG

# 2. Install PyTorch with CUDA 11.8 support
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install PyTorch Geometric and related packages
pip install torch-scatter==latest+cu118 torch-sparse==latest+cu118 torch-cluster==latest+cu118 torch-spline-conv==latest+cu118 -f https://data.pyg.org/whl/torch-2.2.1+cu118.html
pip install torch-geometric==2.3.0

# 4. Install additional dependencies
conda install ninja easydict pyyaml -c conda-forge

# 5. Install compatible Hugging Face packages
# These versions are tested for Python 3.8
pip install transformers==4.46.3 tokenizers==0.20.0 huggingface_hub==0.36.0 safetensors==0.5.3

# 6. Install Sentence Transformers and Datasets
pip install "sentence-transformers[train]==3.0.1" datasets==2.14.7
```

## 2) Download Datasets and Checkpoints
To run experiments, download the required Knowledge Graph datasets and checkpoints for:
- Relation Retriever : sbert
- Triple Retriever : GNN, MPNet
  
You can download all necessary files from the following Google Drive link:
https://drive.google.com/drive/folders/1BVvQRNTaLdONEeFauZfxPYQXQSpCVuNm?usp=drive_link

### 📂 Directory Structure After Download

After downloading, the files should be placed as follows:
```
data/
└── webqsp/
    ├── total_graph_webqsp.jsonl
    ├── webqsp_topic_graph.pkl
    ├── train_goldenpath.jsonl
    └── webqsp_triple2id.pkl
└── cwq/
    ├── total_graph_cwq.jsonl
    ├── cwq_topic_graph.pkl
    ├── train_goldenpath.jsonl
    └── cwq_triple2id.pkl

ckpt/
├── GNN/
│   ├── webqsp_GNN.pth
│   └── webqsp_GNN.pth
├── mpnet/
│   ├── webqsp.mdl
│   └── cwq.mdl
└── sbert/
```
  
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


## 3) Run
```
python main.py --dataset [DATASET_NAME]
```
Replace [DATASET_NAME] with the name of the dataset you want to use (e.g., webqsp, cwq, etc.).
