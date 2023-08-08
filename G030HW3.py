from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    # print(inputPoints.collect())
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")

def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out

def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res

def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)




def MR_kCenterOutliers(points, k, z, L):


    #------------- ROUND 1 ---------------------------
    start = time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    end = time.time()
    print("Time taken by Round 1: ", str((end - start) * 1000), " ms")

    # END OF ROUND 1


    #------------- ROUND 2 ---------------------------

    elems = coreset.collect()
    start = time.time()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    res = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    end = time.time()
    print("Time taken by Round 2: ", str((end - start) * 1000), " ms")
    return res
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def findingMinDist (points, bound):
    minDist = math.inf
    for i in range(0, bound):
        if i < bound:
            for j in range(i + 1, bound):
                dis = squaredEuclidean(points[i],points[j])

                if dis < minDist:
                    minDist = dis
    return math.sqrt(minDist)
def SeqWeightedOutliers (points, weights, k, z, alpha):

    # Initializing the radius
    r = findingMinDist(points,k+z+1) / 2
    guessno = 1
    # print(f"Initial guess = {r}")
    print("Initial guess= ",r)
    while True:
        Z =points.copy()
        # Z = list(set(points.copy()))
        S = []
        Wz = sum(weights)
        tempweight = weights.copy()

        while len(S) < k and Wz > 0:
            # tempweight = weights.copy()
            maxweight = 0
            for x in Z:
                Bz = []
                ball_weight = 0
                for y in Z:

                    if euclidean(x, y) < ((1 + 2 * alpha) * r):
                        Bz.append(y)
                        ball_weight = ball_weight + tempweight[Z.index(y)]
                if ball_weight > maxweight:
                    maxweight = ball_weight
                    newCenter = x

            S.append(newCenter)
            ballwith3r = []

            tempvalues = []
            for y in Z:
                if euclidean(newCenter, y) < ((3 + 4 * alpha) * r):
                    ballwith3r.append(y)
                    Wz = Wz - tempweight[Z.index(y)]
                    tempvalues.append(y)
            for temp in tempvalues:
                del tempweight[Z.index(temp)]
                Z.remove(temp)

        if Wz <= z:

            print("Final guess= ", r)
            # print(f"Number of guesses= {guessno}")
            print("Number of guesses= ", guessno)
            # print(f"Centres are: {S}")
            return S
        else:
            guessno = guessno + 1
            r = 2 * r
#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2 // done!!
#


def computeObjective(points, centers, z):

    ########################################
    def findDist(point):

        mindist = math.inf
        for c in centers:
            dis = squaredEuclidean(point, c)
            if dis < mindist:
                mindist = dis
        return mindist
    distanceToCenter = points.map(lambda point : findDist(point)).top(z+1)
    #sortedDist = distanceToCenter.sortBy(lambda x : x, ascending= True).collect()
    # return math.sqrt(sortedDist[len(sortedDist)-z-1])
    return math.sqrt(distanceToCenter[-1])

        

if __name__ == "__main__":
    main()
