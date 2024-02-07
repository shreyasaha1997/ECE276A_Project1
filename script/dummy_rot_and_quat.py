import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import transforms3d as t3d


R = np.array([[np.cos(np.pi/6), 0, np.sin(np.pi/6)],[0,1,0],[-np.sin(np.pi/6), 0, np.cos(np.pi/6) ]])
quaternion = t3d.quaternions.mat2quat(R)
print(quaternion)
print(R)