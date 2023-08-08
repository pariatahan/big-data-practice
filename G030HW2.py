import math
import sys
import time
import numpy as np


# set of points P
# set of weights W
# number of centers k
# number of outliers z
# coefficient alpha


def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return math.sqrt(res)


def SeqWeightedOutliers(P, W, k, z, alpha):
    # Finding the minimum distance between k+z+1 points
    minDis = math.inf
    for i in range(0, k + z + 1):
        if i < k + z + 1:
            for j in range(i + 1, k + z + 1):
                dis = euclidean(P[i], P[j])

                if dis < minDis:
                    minDis = dis

    # Initializing the radius
    r = minDis / 2
    guessno = 1
    print(f"Initial guess = {r}")
    while True:
        Z = P.copy()
        S = []
        Wz = sum(W)

        while len(S) < k and Wz > 0:
            tempweight = W.copy()
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

            print(f"Final guess= {r}")
            print(f"Number of guesses= {guessno}")

            return S
        else:

            guessno = guessno + 1
            r = 2 * r


def ComputeObjective(P, S, z):
    points = np.array(P)

    centers = np.array(S)

    AllDist = []
    AllCenter = []

    for i in range(0, len(points)):
        AllDist.clear()
        for j in range(0, len(centers)):
            dis = euclidean(points[i], centers[j])
            AllDist.append(dis)

        minDis = min(AllDist)
        Center = AllDist.index(min(AllDist))
        AllCenter.append((Center, minDis))
    NewDist = sorted(AllCenter, key=lambda x: x[1])

    for i in range(z):
        del NewDist[-1]

    NewList = np.array(NewDist)

    finalDist = []
    finalList = []

    for i in range(0, len(centers)):
        finalDist.clear()
        for j in range(0, len(points) - z):
            if NewList[j, 0] == i:
                finalDist.append(NewList[j, 1])
        maximumDis = max(finalDist)
        finalList.append(maximumDis)
    return max(finalList)


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


# Reading the points
inputPoints = readVectorsSeq(sys.argv[1])
print(f"Input size n = {len(inputPoints)}")
# Initializing the weights
weights = [1 for point in inputPoints]
k = int(sys.argv[2])
print(f"Number of centers k = {k}")
z = int(sys.argv[3])
print(f"Number of outliers z = {z}")
start = time.time()
solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
end = time.time()
objective = ComputeObjective(inputPoints, solution, z)
print("Objective function = ", objective)
print(f"Time of SeqWeightedOutliers = {((end - start) * 1000)}")
