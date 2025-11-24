Relation Embeddings for GNN (Entity Scorer)

The Entity Scorer module requires pretrained **relation embedding files**.

| Dataset | File                  |
| ------- | --------------------- |
| WebQSP  | `webqsp/relation.pth` |
| CWQ     | `cwq/relation.pth`    |



#### **Alternatively, Generate Manually**

You can generate the relation embeddings using the GNN module:

```
python3 GNN/get_emb.py --dataset [webqsp] --graph_file data/webqsp/total_graph_webqsp.jsonl
```

