#Autoři: Denis Bastián, Natalie Sokalská

"""
Prohlášení: Návhr skriptu a algoritmu k tvorbě octree byl vytvořen autorem.
AI (ChatGPT - GPT-6, Google Gemini - Hydrogen) byla použita k pomoci při hledání chyb,
nedostatků a návrhu implementace výpočtu a vizualiace křivosti a hustoty bodového mračna.
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
    density = np.full(len(X), np.nan)
    distances = np.full(len(X), np.nan)
    points = np.column_stack((X, Y, Z))

    for point_id in range(len(X)):
        #Get hash of the current point's voxel
        h = idxs[point_id]

        #Get points only from the current voxel
        candidates = voxel_dict[h]

        #Remove the current point
        candidates = [i for i in candidates if i != point_id]

        if len(candidates) == 0:
            continue

        #Calculate distances
        candidate_points = points[candidates]
        query_point = points[point_id]
        dist = np.linalg.norm(candidate_points - query_point, axis=1)

        #Find nearest neighbor
        d_min = np.min(dist)
        distances[point_id] = d_min

        #Density = 1 / d_min^3
        if d_min > 0:
            density[point_id] = 1 / (d_min ** 3)

    #Average nearest-neighbor distance
    if np.any(~np.isnan(distances)):
        d_aver = np.nanmean(distances)
    else:
        d_aver = np.nan

    return density, distances, d_aver


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
print("\nCalculating curvature...")
curvature = computeCurvature(X, Y, Z, boxIndexes, voxel_dict, knn=30)
print("\nCalculating density...")
density, distances, d_aver = computeDensity(X, Y, Z, boxIndexes, voxel_dict)


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
print("======================================")

#Visualization of curvature
drawCurvature(X, Y, Z, curvature)