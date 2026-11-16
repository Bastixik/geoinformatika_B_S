
#Autoři: Denis Bastián, Natalie Sokalská

"""
Prohlášení: Návrh skriptu a algoritmu k tvorbě octree byl vytvořen autory.
AI (ChatGPT - GPT-6, Google Gemini - Hydrogen) byla použita k pomoci při hledání chyb,
nedostatků a návrhu implementace výpočtu a vizualizace křivosti a hustoty bodového mračna.
"""


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


INPUT_FILE = r"C:\Users\denis\Desktop\škola\mgr\geoinformatika\ukol\tree_18.txt"
MAX_POINTS = 50


#Loading file
def loadPoints(file):
    X, Y, Z = [], [], []
    with open(file, "r") as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 3:
                continue
            try:
                X.append(float(values[0]))
                Y.append(float(values[1]))
                Z.append(float(values[2]))
            except ValueError:
                continue

    return X, Y, Z


#Split boxes into 8 smaller boxes
def splitBox(box):
    xmin, ymin, zmin, xmax, ymax, zmax = box

    #Finding midpoint
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    zmid = (zmin + zmax) / 2

    #All possible combinations of min and max values for each dimension to create 8 smaller boxes
    boxes = [
        (xmin, ymin, zmin, xmid, ymid, zmid),
        (xmid, ymin, zmin, xmax, ymid, zmid),
        (xmin, ymid, zmin, xmid, ymax, zmid),
        (xmid, ymid, zmin, xmax, ymax, zmid),
        (xmin, ymin, zmid, xmid, ymid, zmax),
        (xmid, ymin, zmid, xmax, ymid, zmax),
        (xmin, ymid, zmid, xmid, ymax, zmax),
        (xmid, ymid, zmid, xmax, ymax, zmax)
    ]

    return boxes


#Check if a point is inside a box
def pointInBox(point, box):
    x, y, z = point[1], point[2], point[3]
    xmin, ymin, zmin, xmax, ymax, zmax = box
    return (xmin <= x < xmax and ymin <= y < ymax and zmin <= z < zmax)


#Octree implementation
def createOctree(points):

    #Bounding box of all points
    xmin = min(p[1] for p in points)
    xmax = max(p[1] for p in points)

    ymin = min(p[2] for p in points)
    ymax = max(p[2] for p in points)

    zmin = min(p[3] for p in points)
    zmax = max(p[3] for p in points)

    #Slightly enlarge the box
    epsilon = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0) * 0.000001
    rootBox = (xmin - epsilon, ymin - epsilon, zmin - epsilon, xmax + epsilon, ymax + epsilon, zmax + epsilon)

    #Box + points + depth
    boxes = [(rootBox, points, 0)]
    finishedBoxes = []
    numberOfSplits = 0
    maxDepth = 0

    #Subdividing boxes - it divides boxes until each box contains at most MAX_POINTS points
    while boxes:
        box, boxPoints, depth = boxes.pop()
        maxDepth = max(maxDepth, depth)

        #Check if the box contains more points than MAX_POINTS
        if len(boxPoints) <= MAX_POINTS:
            finishedBoxes.append((box, boxPoints))
            continue

        #If there are more points than MAX_POINTS, split the box into 8 smaller boxes
        newBoxes = splitBox(box)
        numberOfSplits += 1

        #Allocate points to the new boxes
        for newBox in newBoxes:
            newPoints = []

            for point in boxPoints:
                if pointInBox(point, newBox):
                    newPoints.append(point)

            if len(newPoints) > 0:
                boxes.append((newBox, newPoints, depth + 1))

    return finishedBoxes, numberOfSplits, maxDepth


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


#Density
def computeDensity(X, Y, Z, idxs, voxel_dict):
    d_sum = 0

    for i in range(len(X)):
        h = idxs[i]
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

    d_aver = d_sum / len(X)
    rho = 1 / d_aver**3

    return d_aver, rho


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
    ax.set_title("Point cloud curvature – Octree")
    plt.tight_layout()
    plt.show()

# Benchmark - calculation time depending on input size
def benchmarkOctree(X, Y, Z, repeats=3):
    n = len(X)

    # Fixed input sizes
    sample_sizes = [100, 1000, 5000, 10000, 25000]

    # Use only sizes that are available in the input data
    sample_sizes = [size for size in sample_sizes if size <= n]

    rng = np.random.default_rng(42)

    mean_times = []

    for size in sample_sizes:
        run_times = []

        for repeat in range(repeats):

            # Randomly select points
            sample_indices = rng.choice(n, size=size, replace=False)

            sample_points = [
                (i, X[index], Y[index], Z[index])
                for i, index in enumerate(sample_indices)
            ]

            # Measure Octree construction time
            startTime = time.process_time()

            createOctree(sample_points)

            runTime = time.process_time() - startTime

            run_times.append(runTime)

        mean_time = np.mean(run_times)
        mean_times.append(mean_time)

    return sample_sizes, mean_times


# Visualization of calculation time
def drawRuntime(sample_sizes, mean_times):
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        sample_sizes,
        mean_times,
        fmt="o-",
        capsize=4
    )

    plt.xlabel("Number of input points")
    plt.ylabel("Calculation time (seconds)")
    plt.title("Octree calculation time depending on input size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Loading the file with points
print("Loading points...")
X, Y, Z = loadPoints(INPUT_FILE)
n = len(X)
print("Number of points:", n)
if n == 0:
    raise ValueError("No points loaded.")


#List of points
points = []
for i in range(n):
    points.append((i, X[i], Y[i], Z[i]))


#Building the octree, processing time is measured
print("\nBuilding Octree...")
startTime = time.process_time()
octree, numberOfSplits, maxDepth = createOctree(points)
buildTime = time.process_time() - startTime


#Generating indexes for final boxes
boxIndexes = np.zeros(n, dtype=int)
voxel_dict = {}

for box_id, (box, boxPoints) in enumerate(octree):
    voxel_dict[box_id] = []

    for point in boxPoints:
        point_id = point[0]
        boxIndexes[point_id] = box_id
        voxel_dict[box_id].append(point_id)


#Calculating curvature and density
print("\nCalculating curvature and density...")
curvature = computeCurvature(X, Y, Z, boxIndexes, voxel_dict, knn=30)

#Calculating density
d_aver, rho = computeDensity(X, Y, Z, boxIndexes, voxel_dict)
print("Density:", rho)


#Printing the results
numberOfBoxes = len(octree)

print("\n======================================")
print("OCTREE RESULTS")
print("======================================")
print("Number of final boxes:", numberOfBoxes)
print("Number of splits:", numberOfSplits)
print("Tree depth:", maxDepth)
print("Run time:", buildTime, "seconds")
print("Average nearest-neighbor distance:", d_aver)

#Benchmark - calculation time depending on input size
print("\nBenchmarking Octree...")

sample_sizes, mean_times = benchmarkOctree(
    X, Y, Z, repeats=3
)

#Visualization of calculation time
drawRuntime(sample_sizes, mean_times)

#Visualization of curvature
drawCurvature(X, Y, Z, curvature)