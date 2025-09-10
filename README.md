# Graph Neural Networks for Hardness Estimation in Robotic Tactile Palpation

This repository contains the code developed for my MSc project on **Graph Neural Networks for Hardness Estimation in Robotic Tactile Palpation**.  
The project explores the use of **vision-based tactile sensing** with the **ViTacTip sensor** and **Graph Neural Networks (GNNs)** to estimate material hardness.  
The ultimate goal is to apply this pipeline for **localising stiff regions in tissue** as a step toward tumour detection.

---

## 📂 Repository Structure
```
tactile-gnn-hardness/
├── model/ # GNN architectures and training scripts
│ ├── graph_outputs
│ │ └── graphs # .pt sample graphs to train and validate the model
│ ├── GINEConv.py # GINE model with edge attributes
│ ├── run_model.py # Train/validate/test GNN models
│ └── vis.py # Visualisation and plotting tools
│
├── tracker/ # Marker tracking algorithms for ViTacTip
│ ├── track_markers_hex.py # For circular sensor layouts
│ ├── track_markers_circ.py # For hexagonal sensor layouts
│ ├── lib_circ.py # Helper functions for circular layout 
│ └── lib_hex.py # Helper functions for hexagonal layout 
│
├── build_graphs/ # Dataset construction
│ ├── build_graph_dataset.py # Convert tracked markers into graph data (.pt)
│ ├── build_graph_dataset_knn.py # Convert tracked markers into graph data using kNN (.pt)
│ └── build_graph_dataset_qc.py # Convert tracked markers into graph data eliminating distant edges (.pt)
│
├── sample_data/ # Sample dataset with 5 cube videos for marker tracking
│ ├── stiffness-cube-1.mvk
│ ├── stiffness-cube-2.mvk
│ ├── stiffness-cube-3.mvk
│ ├── stiffness-cube-4.mvk
│ └── stiffness-cube-5.mvk

```
## 📊 Results

1) GINEConv achieved R² = 0.69, outperforming both GCN and CNN+LSTM baselines.

2) The model effectively captures the relationship between deformation fields and material hardness.

3) This approach provides a proof of concept for applying GNNs to infer physical properties which can be useful for tumour detection.
