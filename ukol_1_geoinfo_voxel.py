
#Autoři: Denis Bastián, Natalie Sokalská

"""
Prohlášení: Hlavní algoritmus k tvorbě voxelizace byl vytvořen autory s využitím souboru "spatial_index.pdf".
AI (ChatGPT - GPT-6, Google Gemini - Hydrogen) byla použita k pomoci při hledání chyb,
nedostatků a návrhu implementace výpočtu a vizualizace křivosti a hustoty bodového mračna.
"""


import math
from time import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


#Loading points from a file (path in main program)
def loadPoints(file):
    X, Y, Z = [], [], []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) == 3:
                x, y, z = parts
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))

    return X, Y, Z


#HOMEWORK IMPLEMENTATION

#Calculate grid size
def calculateGridSize(n):
    k = 3

    n_b = n ** (1 / k)
    n_r = round(n_b ** (1 / k))
    n_r = max(1, n_r)

    return n_b, n_r


#Initialize spatial index
def init_spat_index(X, Y, Z, n_r):
    # Min and max coordinates
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)
    z_min, z_max = min(Z), max(Z)

    #Cell size - for each dimension
    dx = (x_max - x_min) / n_r
    dy = (y_max - y_min) / n_r
    dz = (z_max - z_min) / n_r

    return dx, dy, dz, x_min, y_min, z_min


#Compute spatial indexes ix, iy, iz
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
    #Each key represents a voxel and contains point indices
    voxel_dict = defaultdict(list)

    for i in range(len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        h = hash(ix, iy, iz, n_r)
        voxel_dict[h].append(i)

    return voxel_dict


#Compute and store the 1D hash index of each point's 3D voxel
def createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r):
    idxs = []

    for i in range(len(X)):
        ix, iy, iz = get3Dindex(X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        h = hash(ix, iy, iz, n_r)
        idxs.append(h)

    return idxs


#Density
def computeDensity(X, Y, Z, idxs, voxel_dict):
    d_sum = 0

    for i in range(len(X)):
        #Get hash of the current point's voxel
        h = idxs[i]

        #Get points only from the current voxel
        candidates = voxel_dict[h]

        #Remove the current point
        candidates = [j for j in candidates if j != i]

        if len(candidates) == 0:
            continue

        d_min = np.inf

        for j in candidates:
            distance = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)

            if distance < d_min:
                d_min = distance

        d_sum = d_sum + d_min

    #Average nearest-neighbor distance
    d_aver = d_sum / len(X)

    #Overall density
    rho = 1 / d_aver**3

    return d_aver, rho


#Calculate curvature
def computeCurvature(X, Y, Z, idxs, voxel_dict, knn=30):
    curvature = np.zeros(len(X))
    points = np.column_stack((X, Y, Z))

    for point_id in range(len(X)):
        #Get hash of the current point's voxel
        h = idxs[point_id]

        #Get points only from the current voxel
        candidates = voxel_dict[h]

        #Remove the current point
        candidates = [i for i in candidates if i != point_id]

        #At least 3 points are needed for PCA
        if len(candidates) < 3:
            curvature[point_id] = 0.0
            continue

        #Calculate distances to candidate points
        candidate_points = points[candidates]
        query_point = points[point_id]
        dist = np.linalg.norm(candidate_points - query_point, axis=1)

        #Select up to knn nearest neighbors
        n_neighbors = min(knn, len(candidates))
        nearest_positions = np.argpartition(dist, n_neighbors - 1)[:n_neighbors]
        nearest_indices = np.array(candidates)[nearest_positions]

        #Coordinates of nearest neighbors
        nearest_points = points[nearest_indices]

        #Calculate centroid
        centroid = np.mean(nearest_points, axis=0)

        #Center coordinates
        centered = nearest_points - centroid

        #Covariance matrix
        covariance_matrix = np.cov(centered, rowvar=False)

        try:
            #Eigenvalues of covariance matrix
            eigenvalues = np.linalg.eigvalsh(covariance_matrix)
            eigenvalues = np.sort(eigenvalues)
            eigenvalue_sum = np.sum(eigenvalues)

            if eigenvalue_sum > 0:
                #Curvature = smallest eigenvalue / sum
                curvature[point_id] = eigenvalues[0] / eigenvalue_sum
            else:
                curvature[point_id] = 0.0

        except np.linalg.LinAlgError:
            curvature[point_id] = 0.0

    return curvature


#Visualization of curvature
def drawCurvature(X, Y, Z, curvature):
    tree_cmap = LinearSegmentedColormap.from_list("tree", ["#371E0F", "#6E542C", "#B5A94F", "#73E34A", "#00A63DFF"])
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X, Y, Z, c=curvature, cmap=tree_cmap, s=2)
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    colorbar.set_label("Curvature (κ)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point cloud curvature – voxelization")
    plt.tight_layout()
    plt.show()


#Starting the voxelization
def runVoxelization(X, Y, Z, n_r):
    start = time()

    #Initialize spatial index
    dx, dy, dz, x_min, y_min, z_min = init_spat_index(X, Y, Z, n_r)

    #Create voxel dictionary
    voxel_dict = voxelize(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)

    #Create spatial indexes
    idxs = createSpatialIDXs(X, Y, Z, dx, dy, dz, x_min, y_min, z_min, n_r)

    #Compute density
    d_aver, rho = computeDensity(X, Y, Z, idxs, voxel_dict)

    #Compute curvature
    curvature = computeCurvature(X, Y, Z, idxs, voxel_dict, knn=30)
    elapsed_time = time() - start

    return d_aver, rho, curvature, elapsed_time


#Benchmarking functions - number of points
def benchmarkPointCount(X, Y, Z):
    test_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, len(X)]
    test_sizes = [size for size in test_sizes if size <= len(X)]

    computation_times = []
    rng = np.random.default_rng(42)

    #Take random samples of points and run voxelization for each sample size
    for size in test_sizes:
        print("\nTesting number of points:", size)

        if size == len(X):
            indices = np.arange(len(X))
        else:
            indices = np.sort(rng.choice(len(X), size=size, replace=False))

        X_test = np.asarray(X)[indices].tolist()
        Y_test = np.asarray(Y)[indices].tolist()
        Z_test = np.asarray(Z)[indices].tolist()

        #Calculate grid size separately for each sample
        n_b, n_r = calculateGridSize(size)

        #Run voxelization and measure computation time
        result = runVoxelization(X_test, Y_test, Z_test, n_r)
        computation_times.append(result[3])

    #Visualization of computation time vs number of points
    plt.figure(figsize=(9, 6))
    plt.plot(test_sizes, computation_times, marker="o")
    plt.xlabel("Number of points")
    plt.ylabel("Computation time (s)")
    plt.title("Voxelization – computation time vs number of points")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return test_sizes, computation_times


#Benchmarking function for grid size
def benchmarkGridSize(X, Y, Z):
    grid_sizes = [5, 10, 15, 20, 25, 30]
    computation_times = []

    #Run voxelization for each grid size and measure computation time
    for n_r_test in grid_sizes:
        print("\nTesting grid size:", n_r_test)

        result = runVoxelization(X, Y, Z, n_r_test)
        computation_times.append(result[3])

    #Visualization of computation time vs grid size
    plt.figure(figsize=(9, 6))
    plt.plot(grid_sizes, computation_times, marker="o")
    plt.xlabel("Number of voxels per axis (n_r)")
    plt.ylabel("Computation time (s)")
    plt.title("Voxelization – computation time vs grid size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return grid_sizes, computation_times

#MAIN PROGRAM

file = r"C:\Users\denis\Desktop\škola\mgr\geoinformatika\geoinformatika_B_S\tree_18.txt"

#Load points
X, Y, Z = loadPoints(file)

#Calculate grid size
n_b, n_r = calculateGridSize(len(X))

#Run voxelization
d_aver, rho, curvature, elapsed_time = runVoxelization(X, Y, Z, n_r)

#Print results
print("\nVoxelization results:")
print("Average nearest-neighbor distance:", d_aver)
print("Density:", rho)
print("Average curvature:", np.mean(curvature))
print("Runtime:", elapsed_time, "seconds")

#Visualize curvature
drawCurvature(X, Y, Z, curvature)

#Running benchmarks
benchmarkPointCount(X, Y, Z)
benchmarkGridSize(X, Y, Z)