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
        reader = csv.reader(f, delimiter=',')
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
# data_path = sys.argv[1] # minus  div


T_c2b = np.matrix(np.identity( 4 ))
# camera coordinate systems following a **right-handed** convention
T_c2b[0:4,0:4] = np.array([[0.0,-1.0,0.0,-0.028],[1.0,0.0,0.0,-0.020],[0.0, 0.0, 1.0,0.0],[0.0,0.0,0.0,1.0]])
print T_c2b



# data_path = '/Volumes/BlackSSD/11_21_2020/ip_11_21/2020-11-21T23-49-23/ARposes.txt'

data_path =  sys.argv[1]
insertIndex = len(data_path)-4;
# output_line = data_path[:insertIndex] + '_arkit' + data_path[insertIndex:]
output_line =  sys.argv[2]
# output_line = "/Users/jin/data/HeZhang/VCU_RVI_dataset/robot/room/hard/results/vins_ca/est_lab1.csv"

data = dataFromCSV(data_path)
# data[:,0] = data[:,0]/1e9

timeIndex = data[:,0];

startTime = data[0,0] #1.606020607632468e+09 -2.0

# data[0,0]# 3584619707428/1e9 
startIdx = find_nearest(timeIndex, startTime)

print ("skip: ", (data[startIdx,0] -data[0,0]))
endIdx = find_nearest(timeIndex, startTime+300000)

data = data[startIdx:endIdx,0:8]
# data[:,0] = (data[:,0]-data[0,0])

# TUM: 'timestamp tx ty tz qx qy qz qw' 
# vins:w x y z'

quatXYZ = np.copy(data[:,5:8])
quatW = np.copy(data[:,4])
data[:,4:7]= quatXYZ
data[:,7]= quatW


length =  data.shape[0]
for i in range(length):
    T_c02ci = np.matrix(np.identity( 4 ))
    t = data[i,1:4]
    T_c02ci[0:3,3] = t.reshape((3, 1))
    r = R.from_quat(data[i,4:8])
    T_c02ci[0:3,0:3] = r.as_dcm();
    T_c02bi = T_c02ci*T_c2b

    if (i==0):
        T_bs2c0 = np.linalg.inv(T_c02bi) 

    T_bs2bi = T_bs2c0*T_c02bi;

    data[i,1:4]= (T_bs2bi[0:3,3]).reshape(3)
    r = R.from_dcm( T_bs2bi[0:3,0:3] )
    quat = r.as_quat()
    data[i,4:8]= quat


print output_line
np.savetxt(output_line, data, delimiter=' ', fmt=('%10.9f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f', '%2.6f'))
