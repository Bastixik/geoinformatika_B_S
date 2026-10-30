import math
from time import time
from collections import defaultdict
import numpy as np

#Loading points from a file (path in main program)
def loadPoints(file):
    X, Y, Z = [], [], []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                x, y, z = parts
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))
    return X, Y, Z

#Nearest neighbor search - using brute-force
def getNN(xq, yq, zq, X, Y, Z):
    dmin = math.inf
    xn, yn, zn = X[0], Y[0], Z[0]
    
    for i in range(len(X)):
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        
        if 0 < d < dmin:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
            
    return xn, yn, zn, dmin

#HOMEWORK IMPLEMENTATION
#Initialize spatial index
def init_spat_index(X, Y, Z, n_r):
    #min and max coordinates
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)
    z_min, z_max = min(Z), max(Z)
    
    #Cell size - for each dimension
    dx = (x_max - x_min) / n_r
    dy = (y_max - y_min) / n_r
    dz = (z_max - z_min) / n_r
    
    return dx, dy, dz, x_min, y_min, z_min

# Compute spatial indexes ix, iy, iz
def get3Dindex(x, y, z, dx, dy, dz, x_min, y_min, z_min, n_r):
    #Multiplication by n_r is necessary to cancel the n_r in the dx variable so it matches the calculation in the PDF
    c = 0.99
    x_prime = (x - x_min) / (dx * n_r)
    y_prime = (y - y_min) / (dy * n_r)
    z_prime = (z - z_min) / (dz * n_r)

    #Searching for point's index in each coordinate direction
    ix = int(c * n_r * x_prime)
    iy = int(c * n_r * y_prime)
    iz = int(c * n_r * z_prime)
    
    return ix, iy, iz

#Generate 1D hash from 3D indices
def hash(ix, iy, iz, n_r):
    return ix + iy * n_r + iz * (n_r ** 2)

#Voxelize dataset
def voxelize(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r):
    #defaultdict creates keys (representing voxels) and each key contains a list of point indices that fall into that voxel
    voxel_dict = defaultdict(list)
    #creates index, then hash and finally it appends the point to the corresponding voxel - it runs for the number of X coordinates in the dataset
    for i in range(len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        h = hash(ix, iy, iz, n_r)
        voxel_dict[h].append(i)
        
    return voxel_dict

#Compute and store the 1D hash index of each point's 3D voxel - Changed from the original comment
def createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r):
    idxs = []
    #creates a list of 1D hashes for each point (based on the number of coordinates X in the dataset)
    for i in range(len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        h = hash(ix, iy, iz, n_r)
        idxs.append(h)
        
    return idxs

# Compute spatial density for each voxel
def computeDensity(X, Y, Z, idxs):
    hash_counts = defaultdict(int)

    #uses h from previous function - if h repeats, it increases the count points in voxel
    for h in idxs:
        hash_counts[h] += 1

    #ascending order of the voxels and prints the number of points in each voxel
    for h, count in sorted(hash_counts.items()):
        print(f"Buňka {h}: {count} bodů")

    return [hash_counts[h] for h in idxs]

# Compute curvature for each point - not sure about this - help of Gemini
def computeCurvature(X, Y, Z, idxs):
    voxel_points = defaultdict(list)
    for i, h in enumerate(idxs):
        voxel_points[h].append(i)
        
    curvature = [0.0] * len(X)
    
    for h, indices in voxel_points.items():
        if len(indices) < 3:
            continue
            
        pts = np.array([[X[i], Y[i], Z[i]] for i in indices])
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        cov_matrix = np.cov(centered, rowvar=False)
        
        try:
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)
            e_sum = np.sum(eigenvalues)
            c = eigenvalues[0] / e_sum if e_sum > 0 else 0.0
        except np.linalg.LinAlgError:
            c = 0.0
            
        for i in indices:
            curvature[i] = c
            
    return curvature

#MAIN PROGRAM
if __name__ == '__main__':
    #Loading points from the file
    X, Y, Z = loadPoints('tree_18.txt')
    n = len(X)
    print(f"Načteno bodů: {n}")

    #Grid creation and n_r calculation
    n_r = max(1, int(round(n ** (1 / 9))))
    dx, dy, dz, x_min, y_min, z_min = init_spat_index(X, Y, Z, n_r)
    
    #voxelization of the dataset and two data structures
    voxel_dict = voxelize(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)
    idxs = createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)
    
    #stats
    density = computeDensity(X, Y, Z, idxs)
    curvature = computeCurvature(X, Y, Z, idxs)
    print(f"Atributy hustoty a křivosti úspěšně vypočítány pro {len(density)} bodů.")