import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import math
import sys
import argparse
import time


def d2r(d):
    return d*math.pi/180.


def r2d(r):
    return r*180./math.pi


def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def transformRt(R, t):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and Rotation matrix.

    Input:
    R -- Rotation matrix
    t -- (tx,ty,tz) is the 3D position 

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    return np.array((
        (R[0][0],          R[0][1],         R[0][2], t[0]),
        (R[1][0],          R[1][1],         R[1][2], t[1]),
        (R[2][0],          R[2][1],         R[2][2], t[2]),
        (0.0,                 0.0,                 0.0, 1.0)
    ), dtype=np.float64)


def dataFromCSV(raw_path):
    with open(raw_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        headers = next(reader)
        headers = next(reader)
        headers = next(reader)
        headers = next(reader)
        headers = next(reader)
        headers = next(reader)

        # print(headers)
        # get all the rows as a list
        data = list(reader)

        data = np.array(data)
        data[data == ''] = '0'
        data = data.astype(np.float64)

    return data

# def transformRt(R, t):

#     return np.array((
#     (            R[0][0],          R[0][1],         R[0][2], t[0]),
#     (            R[1][0],          R[1][1],         R[1][2], t[1]),
#     (            R[2][0],          R[2][1],         R[2][2], t[2]),
#     (                0.0,                 0.0,                 0.0, 1.0)
#     ), dtype=np.float64)


def solveRT(p, q):  # from p to q
    m = p.shape[1]
    n = q.shape[1]

    # normalize weights
    weights = np.ones((1, n))
    weights = weights / n

    # find data centroid and deviations from centroid
    q_bar = q.dot(weights.T)
    q_mark = q - np.kron(np.ones((1, n)), q_bar)
    # Apply weights
    q_mark = q_mark * (np.kron(np.ones((3, 1)), weights))

    # find data centroid and deviations from centroid
    p_bar = p.dot(weights.T)
    p_mark = p - np.kron(np.ones((1, m)), p_bar)
    # # Apply weights
    # # p_mark = p_mark .* np.kron(np.ones((3,1)),weights);

    N = p_mark.dot(q_mark.T)   # taking points of q in matched order

    U, d, V = np.linalg.svd(N.T)  # singular value decomposition
    S = np.matrix(np.identity(3))
    detX = np.linalg.det(U) * np.linalg.det(V)
    if (detX < -0.5):
        S[2, 2] = -1

    R = U*S*V

    T = q_bar - R.dot(p_bar)

    # print (R.dot(p)+T - q)

    return R, T


def CleanData(errorT, data):
    [pointsLength, potentialPointsNum] = data.shape
    potentialPointsNum = int((potentialPointsNum-2)/3)

    allPoint = np.zeros((pointsLength, 3, potentialPointsNum))
    for i in range(potentialPointsNum):
        allPoint[:, :, i] = data[:, i*3+2:(i+1)*3+2]

    markers = np.zeros((pointsLength, 3, 6))
    for i in range(pointsLength):
        for j in range(potentialPointsNum):
            ppdistance = np.linalg.norm(
                allPoint[i, :, :] - (allPoint[i, :, j]*(np.ones((potentialPointsNum, 1)))).T, 2, 0)
            # p = [[0.0,0.0,0.0],[0.0,0.099,0],[0.0,-0.045,0],[0.0,-0.115,0],[0.0,0.0,-0.110],[0.0,-0.045,-0.060]]
            check045 = abs(ppdistance - 0.045)
            check075 = abs(ppdistance - 0.075)
            check099 = abs(ppdistance - 0.099)
            check110 = abs(ppdistance - 0.110)
            check115 = abs(ppdistance - 0.115)

            check045_result = np.where(check045 < errorT)
            check075_result = np.where(check075 < errorT)
            check099_result = np.where(check099 < errorT)
            check110_result = np.where(check110 < errorT)
            check115_result = np.where(check115 < errorT)

            # p = [[0,0,0],[0,0.04,0],[0.06,0,0],[-0.03,0,0]]
            if (len(check045_result[0]) and len(check075_result[0]) and len(check099_result[0]) and len(check110_result[0]) and len(check115_result[0])):
                # double check if it misclassifies the laser point
                laserDistance = np.linalg.norm(
                    allPoint[i, :, check045_result[0][0]] - allPoint[i, :, check110_result[0][0]], 2, 0)
                check070 = abs(laserDistance - 0.070)
                check070_result = np.where(check070 < errorT)
                if (len(check045_result[0])):
                    markers[i, :, 0] = allPoint[i, :, j]
                    markers[i, :, 1] = allPoint[i, :, check099_result[0][0]]
                    markers[i, :, 2] = allPoint[i, :, check045_result[0][0]]
                    markers[i, :, 3] = allPoint[i, :, check115_result[0][0]]
                    markers[i, :, 4] = allPoint[i, :, check110_result[0][0]]
                    markers[i, :, 5] = allPoint[i, :, check075_result[0][0]]
                    break
                # if (len(check045_result[0]) )
                # print markers[i,:,:].T
                # break

    cleanData = np.concatenate((markers[:, :, 0], markers[:, :, 1], markers[:, :, 2],
                                markers[:, :, 3], markers[:, :, 4], markers[:, :, 5]), axis=1)

    print(cleanData.shape)
    timestamp = (data[~np.all(cleanData == 0, axis=1)])[:, 1:2]
    # print timestamp[0:3,:]
    # print data[0:3,:]
    cleanData = cleanData[~np.all(cleanData == 0, axis=1)]

    cleanData = np.concatenate((timestamp, cleanData), axis=1)
    print(cleanData.shape)

    return cleanData


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


raw_path = sys.argv[1]
if __name__ == "__main__":

    # raw_path = '/Volumes/BlackSSD/11_21_2020/gt_raw/4min.csv'
    # gt_path = '/Volumes/BlackSSD/test.csv'
    insertIndex = len(raw_path)-4
    gt_path = raw_path[:insertIndex] + '_gt' + raw_path[insertIndex:]

    startTime = 0

    t = time.time()
    data = dataFromCSV(raw_path)
    elapsed = time.time() - t
    print("finished in ", elapsed)

    errorT = 0.002
    data = data[0:99999999, :]

    timeIndex = data[:, 1]
    # startIdx = find_nearest(timeIndex, data[startIdx,1]+0.811)
    startIdx = 1

    startIdx = find_nearest(timeIndex, startTime)
    # print (startIdx)

    endIdx = find_nearest(timeIndex, data[startIdx, 1]+4000)

    # print (endIdx)

    data = data[startIdx:endIdx, :]

    t = time.time()

    cleanData = CleanData(errorT, data)
    elapsed = time.time() - t
    print("finished in ", elapsed)

    cleanData[:, 0] = cleanData[:, 0] - cleanData[0, 0]

    # print( cleanData[1,0] - cleanData[0,0] )
    # print cleanData.shape
    cleanDataSize = cleanData.shape[0]

    t_pre = 0
    roll = 0
    pitch = 0
    yaw = 0
    T_led2b = np.matrix(np.identity(4))
    r = R.from_euler('zyx', [[0,  0,  0]], degrees=True)

    T_led2b[0:3, 0:3] = np.array(np.array(r.as_matrix()))
    # T_led2b[0:3,0:3] = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])

    T_led2b[0, 3] = -0.034
    T_led2b[1, 3] = 0.0
    T_led2b[2, 3] = 0.0

    print(T_led2b)

    T_b2led = np.matrix(np.identity(4))
    T_b2led = np.linalg.inv(T_led2b)

    # p = [[0.0,0.0,0.0],[0.0,0.099,0],[0.0,-0.045,0],[0.0,-0.115,0],[0.0,0.0,-0.110],[0.0,-0.045,-0.060]]
    p = [[0.0, 0.0, 0.0], [0.0, -0.099, 0], [0.0, 0.045, 0],
         [0.0, 0.115, 0], [0.0, 0.0, 0.110], [0.0, 0.045, 0.060]]

    p = np.array(p)
    T = np.zeros((cleanDataSize, 7))
    Ttest = np.zeros((cleanDataSize, 7))
    T_leds2led0 = np.matrix(np.identity(4))
    for i in range(cleanDataSize):
        q = [	[cleanData[i, 1], cleanData[i, 2], cleanData[i, 3]],
              [cleanData[i, 4], cleanData[i, 5], cleanData[i, 6]],
              [cleanData[i, 7], cleanData[i, 8], cleanData[i, 9]],
              [cleanData[i, 10], cleanData[i, 11], cleanData[i, 12]],
              [cleanData[i, 13], cleanData[i, 14], cleanData[i, 15]],
              [cleanData[i, 16], cleanData[i, 17], cleanData[i, 18]]]
        q = np.array(q)
        Rsvd, Tsvd = solveRT(p.T, q.T)

        camPosition = np.array([0.00, 0.00, 0.0])

        Tsvd = (Tsvd.T+(Rsvd).dot(camPosition)).T

        T_led02ledi = np.matrix(np.identity(4))
        T_led02ledi[0:3, 3] = Tsvd
        T_led02ledi[0:3, 0:3] = Rsvd

        if (i == 0):
            T_leds2led0 = np.linalg.inv(T_led02ledi)

        T_bs2bi = T_b2led*T_leds2led0*T_led02ledi*T_led2b

        T[i, 0:3] = (T_bs2bi[0:3, 3]).reshape(3)

        r = R.from_matrix(T_bs2bi[0:3, 0:3])
        quat = r.as_quat()
        T[i, 3:7] = quat

    print("-------------")

    groundTruth = np.concatenate((cleanData[:, 0:1], T), axis=1)

    # gt_path = "/Users/jin/data/HeZhang/VCU_RVI_dataset/robot/room/hard/ground_truth/gt_lab1.csv"
    print(gt_path)

    np.savetxt(gt_path, groundTruth, delimiter=' ', fmt=(
        '%10.9f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f'))
