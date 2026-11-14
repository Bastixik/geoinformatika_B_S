
# METHOD 1: NAIVE (NEAREST NEIGBOUR SEARCH)
# Spatial density computation + computation time 


from time import time
import numpy as np
import matplotlib.pyplot as plt


# Loads points from a file (x, y, z)

def loadPoints(file):
    X, Y, Z = [], [], []

    with open(file) as f:
        for line in f:
            x, y, z = line.split('\t')
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))

    return X, Y, Z



# The function takes one point (xq, yq, zq) and all points X, Y, Z.
# Returns the distance to the nearest of them (dmin)

def getNN(xq, yq, zq, X, Y, Z):
    dmin = np.inf

    for i in range(len(X)):                               # goes through all points
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]      
        d = (dx*dx + dy*dy + dz*dz) ** 0.5                # Pythagorean theorem

        if 0 < d < dmin:                                  # the point skips itself
            dmin = d

    return dmin



# Point cloud density: rho = 1 / d_aver^3
# d_aver = average distance to the nearest neighbour 

def computeDensity(X, Y, Z):
    d_sum = 0

    for i in range(len(X)):
        d_sum = d_sum + getNN(X[i], Y[i], Z[i], X, Y, Z)

    d_aver = d_sum / len(X)
    rho = 1 / d_aver**3

    return d_aver, rho



# Runs the density computation several times, each time on a different number of points, and measures the time.
# (random selection of points, last size = all points)

def benchmarkPointCount(X, Y, Z):
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, len(X)]
    times = []
    rng = np.random.default_rng(42)

    for n in sizes:
        idx = np.sort(rng.choice(len(X), size=n, replace=False)) # draws n different numbers, sorts them in ascending order, no number is drawn twice
        Xs = [X[i] for i in idx]
        Ys = [Y[i] for i in idx]
        Zs = [Z[i] for i in idx]

        # Density computation + time measurement
        t_start = time()
        d_aver, rho = computeDensity(Xs, Ys, Zs)
        t = time() - t_start
        times.append(t)

        print("n =", n, "  d_aver =", d_aver, "m,  rho =", rho, ", time =", t, "s")

    # Result of the last run = whole point cloud
    return sizes, times, d_aver, rho



# Plot: computation time as a function of input set size

def drawTimes(sizes, times):
    plt.figure(figsize=(9, 6))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Number of points n")
    plt.ylabel("Computation time (s)")
    plt.title("Naive search – Computation time as a function of point cloud size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# MAIN PROGRAM

# Loading points
X, Y, Z = loadPoints("tree_18.txt")
print("Number of points:", len(X))

# Density + times for different input sizes
sizes, times, d_aver, rho = benchmarkPointCount(X, Y, Z)

# Result for the whole point cloud
print()
print("Result – naive method, all points")
print("Average distance to the nearest neighbour d_aver =", d_aver, "m")
print("Density rho =", rho, "points/m^3")

# Plot of times
drawTimes(sizes, times)