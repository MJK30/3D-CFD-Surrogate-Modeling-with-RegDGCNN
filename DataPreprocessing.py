


import os
import numpy as np
import trimesh

RAW_DIR = "data"
OUT_DIR = "Processed_Data"
os.makedirs(OUT_DIR, exist_ok= True)

def load_ids(filename):
    with open(filename, "r") as f:
        return f.read().strip().split(",")
    
def process_id(id):
    id = id.zfill(3)
    mesh_path = os.path.join(RAW_DIR,f"mesh_{id}.ply")
    pressure_path = os.path.join(RAW_DIR, f"press_{id}.npy")
    output_path = os.path.join(OUT_DIR, f"{id}.npz")
    
    if not (os.path.exists(mesh_path) and os.path.exists(pressure_path)):
        print(f"Missing: {id}")
        return
    
    mesh = trimesh.load(mesh_path)
    points = mesh.vertices.astype(np.float32)
    pressure = np.load(pressure_path).astype(np.float32)
    pressure = pressure[0:3586] 
    # there is a mismatch in the datasets. Pressure has 3682 datapoints; Mesh has 3586 datapoints.
    
    
    if len(points) != len(pressure):
        print(f"⚠️  Mismatch for {id}: {len(points)} points, {len(pressure)} pressure")
        return
    
    np.savez(output_path, points= points, pressure=pressure)
    

def preprocess():
    ids = load_ids("train.txt") + load_ids("test.txt")
    for id in ids:
        process_id(id)
        
    
if __name__ == "__main__":
    preprocess()
        