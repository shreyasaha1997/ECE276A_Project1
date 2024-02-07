import os
import numpy as np

gyroscope_scale_factor = 3300 / 1023 * (np.pi/180) / 3.33
accelerometer_scale_factor = 3300 / 1023 / 300

def get_gyroscope_bias(omegas):
    wx = np.mean(omegas[:100, 0])
    wy = np.mean(omegas[:100, 1])
    wz = np.mean(omegas[:100, 2])
    return wx, wy, wz

def fix_gyroscope_data(omegas):
    omegas = omegas[:, [1, 2, 0]]
    wx, wy, wz = get_gyroscope_bias(omegas)
    for i in range(len(omegas)):
        omegas[i][0] = (omegas[i][0]-wx)*gyroscope_scale_factor
        omegas[i][1] = (omegas[i][1]-wy)*gyroscope_scale_factor
        omegas[i][2] = (omegas[i][2]-wz)*gyroscope_scale_factor
    return omegas

def get_acceleration_bias(accls):
    ax = np.mean(accls[:100, 0])
    ay = np.mean(accls[:100, 1])
    az = [i+(-1/accelerometer_scale_factor) for i in accls[:500, 2]] ##9.8
    az = np.mean(az)
    return ax, ay, az

def fix_acceleration_data(accls):
    for i in range(len(accls)):
        # print(accls[i][2])
        accls[i][0] = -accls[i][0]
        accls[i][1] = -accls[i][1]
    ax, ay, az = get_acceleration_bias(accls)
    for i in range(len(accls)):
        # print(accls[i][2])
        accls[i][0] = (accls[i][0]-ax)*accelerometer_scale_factor*(-9.8)
        accls[i][1] = (accls[i][1]-ay)*accelerometer_scale_factor*(-9.8)
        accls[i][2] = (accls[i][2]-az)*accelerometer_scale_factor*(-9.8)
    return accls

def load_data(args):
    basedir = os.path.join(args.basedir, args.dataset)
    accls = np.load(os.path.join(basedir, 'accl.npy')).astype(float)
    omegas = np.load(os.path.join(basedir, 'omega.npy')).astype(float)
    ts = np.load(os.path.join(basedir, 'ts.npy')).astype(float)

    gt_rots, gt_ts = None, None
    if os.path.exists(os.path.join(basedir, 'vicd_rots.npy')):
        gt_rots = np.load(os.path.join(basedir, 'vicd_rots.npy')).astype(float)
        gt_ts = np.load(os.path.join(basedir, 'vicd_ts.npy'))

    cam_imgs, cam_ts = None, None
    if os.path.exists(os.path.join(basedir, 'images.npy')):
        cam_imgs = np.load(os.path.join(basedir, 'images.npy')).astype(float)
        cam_ts = np.load(os.path.join(basedir, 'cam_ts.npy')).astype(float)

    omegas = fix_gyroscope_data(omegas)
    accls = fix_acceleration_data(accls)
    return accls.astype(float), omegas.astype(float), gt_rots, ts.astype(float), gt_ts, cam_imgs, cam_ts