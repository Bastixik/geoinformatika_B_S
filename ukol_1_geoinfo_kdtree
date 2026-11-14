
# METHOD 3: ACCELERATED SEARCH USING A KD-TREE
# Computation of spatial density + curvature (PCA, knn = 30)
# Curvature visualization + Computation time as a function of point cloud size




from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Loading points from a file (x, y, z separated by a tab)

def loadPoints(file):
    X, Y, Z = [], [], []

    with open(file) as f:
        for line in f:
            x, y, z = line.split('\t')
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))

    return X, Y, Z


# Distance between two points p, q = [x, y, z]
def dist(p, q):
    dx, dy, dz = p[0] - q[0], p[1] - q[1], p[2] - q[2]  # p[0] is x, p[1] is y, p[2] is z
    return (dx*dx + dy*dy + dz*dz) ** 0.5



# KD-TREE NODE
class KDNode:

    def __init__(self, point=None, axis=None, left=None, right=None, point_indices=None):

        self.point = point                    # coordinates of the point in the node [x, y, z]
        self.axis = axis                      # splitting axis: 0 = x, 1 = y, 2 = z
        self.left = left                      # left subtree (smaller coordinates)
        self.right = right                    # right subtree (larger coordinates)
        self.point_indices = point_indices    # index of the point in the node


# TREE
class KDTree:

    def __init__(self, data):

        self.data = np.array(data)                                          # numpy table: n rows and 3 columns (x, y, z)
        self.root = None                                                    # at the beginning the tree has no root
        self.root = self._build_tree(list(range(len(self.data))), 0)        # list of indices of all points, 0 is the initial depth (root)

    # BUILDING THE TREE
    # sort points by axis, median = node, split the rest
    def _build_tree(self, idxs, depth):                 # idxs are indices of the points this part of the tree is built from
        if len(idxs) == 0:
            return None

        axis = depth % 3                                # depth (%) remainder after division: depth 0 gives x, 1 gives y, 2 gives z, 3 x again ...
        idxs.sort(key=lambda i: self.data[i][axis])     # sorts point indices by their coordinate on the current axis
        m = len(idxs) // 2                              # median
        i = idxs[m]                                     # index of the median point

        return KDNode(point=self.data[i].tolist(),
                      axis=axis,
                      left=self._build_tree(idxs[:m], depth + 1),       # builds the left subtree from points before the median, one level deeper
                      right=self._build_tree(idxs[m+1:], depth + 1),    # builds the right subtree from points after the median (m)
                      point_indices=i)                                  # point index


    
    # STARTING THE NEAREST NEIGHBOUR SEARCH for point q
    # Starts the search from the root self.root. -->  returns the distance dmin
    def nearest(self, q):
        best = [np.inf]                       
        self._search_nn(self.root, q, best)
        return best[0]


    # NEAREST NEIGHBOUR SEARCH IN THE TREE
    def _search_nn(self, node, q, best):
        if node is None:
            return              

        # Distance to the point in the current node  
        d = dist(node.point, q)
        if 0 < d < best[0]:                 # If it is smaller than the current best, it is stored.
            best[0] = d           

        # Which side of the splitting plane q belongs to
        # diff is the difference of coordinates on the splitting axis
        # negative means that q is on the left, so the left side is the nearer side (near)
        diff = q[node.axis] - node.point[node.axis]
        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        # The nearer side first
        self._search_nn(near, q, best)

        # abs(diff) is the distance of q from the splitting plane
        # The other side only if the splitting plane is closer than the current dmin
        if abs(diff) < best[0]:
            self._search_nn(far, q, best)

    # STARTING THE k NEAREST NEIGHBOURS SEARCH
    # k nearest neighbours of point q (iq = index of point q) -> returns indices of the neighbours
    def knn(self, q, iq, k):
        best = []                                       # (distance, point index)
        self._search_knn(self.root, q, iq, k, best)
        return [j for d, j in best]                     # takes only the point indices from the pairs and returns them
    
    # k NEAREST NEIGHBOURS SEARCH IN THE TREE
    def _search_knn(self, node, q, iq, k, best):
        if node is None:
            return

        # adds a pair, sorts the list by distance and if there are more than 30 points in the list, removes the farthest one (pop() deletes the last element)
        if node.point_indices != iq:
            d = dist(node.point, q)
            if len(best) < k or d < best[-1][0]:
                best.append((d, node.point_indices))
                best.sort()
                if len(best) > k:
                    best.pop()                


        # the nearer side first, the other side only if we do not have 30 neighbours yet, or the plane is closer than the 30th neighbour
        diff = q[node.axis] - node.point[node.axis]
        if diff < 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        self._search_knn(near, q, iq, k, best)

        # The other side: if I do not have k neighbours yet, or the plane is closer than the k-th neighbour
        if len(best) < k or abs(diff) < best[-1][0]:
            self._search_knn(far, q, iq, k, best)


#-------------------------------------------------------------------------------------------------------------------------------
# CHARACTERISTICS

# DENSITY: rho = 1 / d_aver^3
def computeDensity(tree):
    P = tree.data.tolist()                      # converts the numpy table to a list of points [[x0,y0,z0], [x1,y1,z1], …]
    d_sum = 0

    for i in range(len(P)):
        d_sum = d_sum + tree.nearest(P[i])      # for each point P[i] tree.nearest is called

    d_aver = d_sum / len(P)
    rho = 1 / d_aver**3

    return d_aver, rho


# CURVATURE
# Curvature of one point from its neighbours (PCA): kappa = l1 / (l1 + l2 + l3)
def pcaCurvature(data, idx):        
    A = data[idx]                             # table (coordinates of the neighbours)
    A = A - A.mean(axis=0)                    # shifts the cloud so that its centroid (mean) is at the origin
    C = A.T @ A / len(idx)                    # covariance matrix
    lam = np.linalg.eigvalsh(C)               # eigenvalues (from the smallest)

    return lam[0] / (lam[0] + lam[1] + lam[2])


# CURVATURE OF ALL POINTS
def computeCurvature(tree, k):
    P = tree.data.tolist()
    K = []

    for i in range(len(P)):
        idx = tree.knn(P[i], i, k)
        K.append(pcaCurvature(tree.data, idx))

    return K


# WHOLE COMPUTATION
# : building the tree + density + curvature
def runKDTree(X, Y, Z, k):
    tree = KDTree(list(zip(X, Y, Z)))

    d_aver, rho = computeDensity(tree)
    K = computeCurvature(tree, k)

    return d_aver, rho, K



# ----------------------------------------------------------------------------------------------------------------------

# COMPUTATION TIME DEPENDING ON THE NUMBER OF POINTS
# (random selection of points, last size = all points)

def benchmarkPointCount(X, Y, Z, k):
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, len(X)]
    times = []
    rng = np.random.default_rng(42)

    for n in sizes:
        idx = np.sort(rng.choice(len(X), size=n, replace=False))
        Xs = [X[i] for i in idx]
        Ys = [Y[i] for i in idx]
        Zs = [Z[i] for i in idx]

        t_start = time()
        d_aver, rho, K = runKDTree(Xs, Ys, Zs, k)
        t = time() - t_start
        times.append(t)

        print("n =", n, "  d_aver =", d_aver, "m,  rho =", rho, ", time =", t, "s")

    # Results of the last run = whole point cloud
    return sizes, times, d_aver, rho, K



# VISUALIZATION
# Points coloured by curvature
def drawCurvature(X, Y, Z, K):
    tree_cmap = LinearSegmentedColormap.from_list("tree", ["#371E0F", "#6E542C", "#B5A94F", "#73E34A", "#00A63DFF"])
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=K, cmap=tree_cmap, s=2)
    colorbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    colorbar.set_label('Curvature κ')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point cloud curvature – kd-tree (knn = 30)')
    plt.tight_layout()
    plt.show()


# Computation time vs. number of points
def drawTimes(sizes, times):
    plt.figure(figsize=(9, 6))
    plt.plot(sizes, times, marker='o')
    plt.xlabel('Number of points n')
    plt.ylabel('Computation time [s]')
    plt.title('Kd-tree – Computation time as a function of point cloud size')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

# Loading points
X, Y, Z = loadPoints('tree_18.txt')
print("Number of points:", len(X))

# Computation for different input sizes (last = all points)
knn = 30
sizes, times, d_aver, rho, K = benchmarkPointCount(X, Y, Z, knn)

# Result for the whole point cloud
print()
print("Result – kd-tree, all points")
print("Average distance to the nearest neighbour d_aver =", d_aver, "m")
print("Density rho =", rho, "points/m^3")
print("Curvature min:", min(K), "max:", max(K))

# Visualization
drawCurvature(X, Y, Z, K)
drawTimes(sizes, times)