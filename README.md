# HCAN

Source code for [Neural Networks](https://www.sciencedirect.com/journal/neural-networks) paper: [**Hypergraph contrastive attention networks for hyperedge prediction with negative samples evaluation**](https://doi.org/10.1016/j.neunet.2024.106807) 

**Overview:** Hyperedge prediction aims to predict common relations among multiple nodes that will occur in the future or remain undiscovered in the current hypergraph. It is traditionally modeled as a classification task, which performs hypergraph feature learning and classifies the target samples as either present or absent. However, these approaches involve two issues: (i) in hyperedge feature learning, they fail to measure the influence of nodes on the hyperedges that include them and the neighboring hyperedges, and (ii) in the binary classification task, the quality of the generated negative samples directly impacts the prediction results. To this end, we propose a Hypergraph Contrastive Attention Network (HCAN) model for hyperedge prediction. Inspired by the brain organization, HCAN considers the influence of hyperedges with different orders through the order propagation attention mechanism. It also utilizes the contrastive mechanism to measure the reliability of attention effectively. Furthermore, we design a negative sample generator to produce three different types of negative samples. We evaluate the impact of various negative samples on the model and analyze the problems of binary classification modeling. The effectiveness of HCAN in hyperedge prediction is validated by experimentally comparing 12 baselines on 9 datasets.


### Notes:
- All raw datasets used in the paper are from [here](https://github.com/arbenson/ScHoLP-Data) in the data directory
- Codes for preprocessing raw datasets and generating three negative samples are in utils.py

### Envoriment:
- python==3.8.8
- torch==1.12.0
- torch-geometric==2.2.0
- torch-sparse==0.6.16
- torch-scatter==2.1.0
- torch-metrics==0.10.3
  

### Citation:

```bibtex
@article{WANG2025106807,
title = {Hypergraph contrastive attention networks for hyperedge prediction with negative samples evaluation},
journal = {Neural Networks},
volume = {181},
pages = {106807},
year = {2025},
author = {Junbo Wang and Jianrui Chen and Zhihui Wang and Maoguo Gong}
}
```
