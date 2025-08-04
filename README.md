# 3D-CFD-Surrogate-Modeling-with-RegDGCNN

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

### Preprocessing

All data is converted into `.npz` format containing:
- `points`: N × 3 array of 3D coordinates
- `pressure`: N × 1 array of scalar pressure values

