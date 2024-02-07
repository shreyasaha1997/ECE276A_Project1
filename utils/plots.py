import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
import transforms3d as t3d
import os

def get_euler_angles_from_rotation_matrices(R1, R2):
    relative_rotation = R2 @ np.linalg.inv(R1)
    euler_angles = t3d.euler.mat2euler(relative_rotation, axes='sxyz')
    roll, pitch, yaw = euler_angles
    return roll, pitch, yaw

def get_gt_rpy(args):
    basedir = os.path.join(args.basedir, args.dataset)
    rotation_matrices = np.load(os.path.join(basedir, 'vicd_rots.npy'))
    rolls = []
    pitches = []
    yaws = []
    I = np.eye(3)

    for i in range(len(rotation_matrices)):
        roll, pitch, yaw = get_euler_angles_from_rotation_matrices(I, rotation_matrices[i])
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    return rolls, pitches, yaws

def get_predicted_rpy(qs):
    rotation_matrices = [t3d.quaternions.quat2mat(q) for q in qs]
    rolls = []
    pitches = []
    yaws = []
    I = rotation_matrices[0]

    for i in range(len(qs)):
        roll, pitch, yaw = get_euler_angles_from_rotation_matrices(I, rotation_matrices[i])
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)

    return rolls, pitches, yaws

def plot_rpy_epoch(log_path, qs, ts,epoch, gt_ts, args):
    p_rolls, p_pitches, p_yaws = get_predicted_rpy(qs)

    if gt_ts is not None:
        gt_rolls, gt_pitches, gt_yaws = get_gt_rpy(args)

    figure, axis = plt.subplots(3, 1)
    figure.set_size_inches(25,15)
    axis[0].set_ylim(-4, 4) 
    axis[0].plot(ts, p_rolls, color='red', label='estimations') 
    if gt_ts is not None:
        axis[0].plot(gt_ts, gt_rolls, color='blue', label='GT') 
    axis[0].legend()
    axis[0].set_title("roll") 

    axis[1].set_ylim(-4, 4) 
    axis[1].plot(ts, p_pitches, color='red', label='estimations') 
    if gt_ts is not None:
        axis[1].plot(gt_ts, gt_pitches, color='blue', label='GT') 
    axis[1].legend()
    axis[1].set_title("pitch") 
 
    axis[2].set_ylim(-4, 4) 
    axis[2].plot(ts, p_yaws, color='red', label='estimations') 
    if gt_ts is not None:
        axis[2].plot(gt_ts, gt_yaws, color='blue', label='GT')
    axis[2].legend()
    axis[2].set_title("yaw") 

    plt.savefig(os.path.join(log_path, 'euler_angles_'+str(epoch)+'.png'), bbox_inches='tight')

def plot_rpy(args, qs, ts, gt_ts):
    gt_rolls, gt_pitches, gt_yaws = get_gt_rpy(args)
    p_rolls, p_pitches, p_yaws = get_predicted_rpy(qs)

    figure, axis = plt.subplots(3, 1)
    figure.set_size_inches(25,15)
    axis[0].plot(gt_ts, gt_rolls) 
    axis[0].plot(ts, p_rolls) 
    axis[0].set_title("roll") 

    axis[1].plot(gt_ts, gt_pitches) 
    axis[1].plot(ts, p_pitches) 
    axis[1].set_title("pitch") 

    axis[2].plot(gt_ts, gt_yaws)
    axis[2].plot(ts, p_yaws) 
    axis[2].set_title("yaw") 

    basedir = os.path.join(args.basedir, args.dataset)
    plt.savefig(os.path.join(basedir, 'euler_angles.png'), bbox_inches='tight')
