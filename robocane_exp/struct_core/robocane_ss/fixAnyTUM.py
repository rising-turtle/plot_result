import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


import sys
import argparse



def d2r(d):
    return d*math.pi/180.

def r2d(r):
    return r*180./math.pi

def rotationMatrixToEulerAngles(R):
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
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

def dataFromCSV(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        data = list(reader)

        data = np.array(data)
        data[data == ''] = '0'
        data = data.astype(np.float64)

    return data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
# folderName = sys.argv[1]
# # folderName = 'fcxcy_1'
data_path = sys.argv[1] # minus  div
# data_type = sys.argv[2] # gt vins est
# data_path = '/Users/jin/Downloads/ipad/ARposes_tum.txt'

insertIndex = len(data_path)-4;
output_line = data_path[:insertIndex] + '_cp' + data_path[insertIndex:]
# output_line = "/Users/jin/data/HeZhang/VCU_RVI_dataset/robot/room/hard/results/vins_ca/est_lab1.csv"

data = dataFromCSV(data_path)
# data[:,0] = data[:,0]/1e9

timeIndex = data[:,0];

startTime = float(sys.argv[2]) # 1.604275701036828e+09 -2.65

startIdx = find_nearest(timeIndex, startTime)
print ("start from: ", data[startIdx,0])
print ("skip: ", (data[startIdx,0] -data[0,0]))
endIdx = find_nearest(timeIndex, startTime+300000)

data = data[startIdx:endIdx,0:8]
data[:,0] = (data[:,0]-data[0,0])


length =  data.shape[0]
for i in range(length):
    T_b02bi = np.matrix(np.identity( 4 ))
    t = data[i,1:4]
    T_b02bi[0:3,3] = t.reshape((3, 1))
    r = R.from_quat(data[i,4:8])
    # T_b02bi[0:3,0:3] = r.as_matrix();
    T_b02bi[0:3,0:3] = r.as_dcm();

    if (i==0):
        T_bs2b0 = np.linalg.inv(T_b02bi) 
    T_bs2bi = T_bs2b0*T_b02bi;

    data[i,1:4]= (T_bs2bi[0:3,3]).reshape(3)
    # r = R.from_matrix( T_bs2bi[0:3,0:3] )
    r = R.from_dcm( T_bs2bi[0:3,0:3] )
    quat = r.as_quat()
    data[i,4:8]= quat


print (output_line)
np.savetxt(output_line, data, delimiter=' ', fmt=('%10.9f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f'))
