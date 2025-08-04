# 3D CFD Surrogate Modeling with RegDGCNN

This project implements a Regression Dynamic Graph-Based CNN Model (RegDGCNN) to predict the pressure distribution over 3D geometries using Point Cloud Data. Data combines the mesh details with the scalar pressure fields to train the RegDGCNN model. KNN is used to explore the design landscape to analyse the neighbours and form Dynamic Graph Edge features. Edge Convolutions is used to explore the neighbourhood and extract features for each layer. The RegDGCNN predicts pressure fields for each point cloud.  The model learns directly from CFD simulation data and enables rapid, simulation-free aerodynamic design space exploration.

---

##  Project Highlights

- **Geometry-Aware Learning**: Uses RegDGCNN (Dynamic Graph CNN) for 3D point cloud-based regression.
- **Surrogate Modeling**: Predicts pressure fields as a proxy for CFD simulations.
- **ShapeNet-Car CFD Dataset**: 3D mesh data paired with surface pressure values.
- **Design Space Exploration**: Enables efficient aerodynamic assessment of unseen designs and learn the shape of the object.

---

## Dataset

### Source

- **Dataset**: [Three-dimensional flow over ShapeNet-Car](https://doi.org/10.7910/DVN/L6TYNW) (https://zenodo.org/records/13993629)
- **Files Used**:
  - `mesh_xxx.ply` – 3D mesh geometry (point cloud)
  - `press_xxx.npy` – CFD-generated surface pressure scalar field
  - `train.txt`, `test.txt` – Comma-separated lists of sample IDs
### File Descriptions

- main.py - 	Entry point to start training. Loads data, initializes the model, optimizer, and runs training & evaluation loops.
- DataPreprocessing.py - Converts the raw .ply 3D mesh geometry and .npy scalar pressure fields into .npz format for efficient loading. all .npz files stored Processed_Data folder
- utils/dataset.py - Defines the class ShapeNetCFD - a PyTorch Dataset for loading and preprocessing point cloud and pressure data from .npz files
- train_model.py - Contains functions for one epoch of training and evaluation.
- models/RegDGCNN.py - Defines RegDGCNN class, a Dynamic Graph CNN for Regression handling 3D point cloud data
- utils/knn.py - Designed to explore the K nearest Neighbours for edge exploration
 

