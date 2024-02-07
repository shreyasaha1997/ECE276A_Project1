import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

def get_euler_angles(R1, R2):
    r1 = R.from_matrix(R1)
    r2 = R.from_matrix(R2)
    relative_rotation = r2 * r1.inv()
    euler_angles = relative_rotation.as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler_angles
    return roll, pitch, yaw

rotation_matrices = np.load('data/1/vicd_rots.npy')

rolls = []
pitches = []
yaws = []
ts = np.load('data/1/vicd_ts.npy')
ts_arr = []
I = np.eye(3)

for i in range(len(rotation_matrices)):
    roll, pitch, yaw = get_euler_angles(I, rotation_matrices[i])
    rolls.append(roll)
    pitches.append(pitch)
    yaws.append(yaw)
    ts_arr.append(i)

print(max(rolls))

figure, axis = plt.subplots(3, 1)
figure.set_size_inches(25,15)
axis[0].plot(ts_arr, rolls) 
axis[0].set_title("roll") 

axis[1].plot(ts_arr, pitches) 
axis[1].set_title("pitch") 

axis[2].plot(ts_arr, yaws) 
axis[2].set_title("yaw") 

plt.savefig('euler_angles.png', bbox_inches='tight')