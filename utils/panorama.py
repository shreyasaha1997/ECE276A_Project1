import numpy as np
import transforms3d as t3d
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import math

def find_closest_timestamps(qs, qs_ts, img_ts):
    insertion_points = np.searchsorted(qs_ts, img_ts, side='right')
    insertion_points -= 1
    insertion_points = np.clip(insertion_points, 0, len(qs_ts) - 1)
    max_elements = qs_ts[insertion_points]
    unique_elements, unique_indices = np.unique(max_elements, return_index=True)
    qs = qs[insertion_points]
    return max_elements, qs, unique_indices

def find_closest_unique_timestamps(qs, qs_ts, cam_imgs, img_ts):
    closest_ts, closest_qs, unique_indices = find_closest_timestamps(qs, qs_ts, img_ts)
    # img_ts = img_ts[unique_indices]
    # closest_ts = closest_ts[unique_indices]
    # closest_qs = closest_qs[unique_indices]
    # cam_imgs = cam_imgs[unique_indices]
    diff = [abs(a-b) for a,b in zip(img_ts,closest_ts)]
    indices = [i for i in range(len(diff)) if diff[i]<0.01]
    diff = np.array(diff)[indices]
    return closest_ts[indices], closest_qs[indices], cam_imgs[indices]

def get_spherical_coordinates_for_image(img,hpr=60,vert=45):
    w = img.shape[1]
    h = img.shape[0]
    del_alpha = float(hpr/w)
    del_beta = float(vert/h)
    spherical_coordinates = {}
    for i in range(w):
        for j in range(h):
            rgb = img[j][i]
            sph_coords = ((i-w/2)*del_alpha, (j-h/2)*del_beta, 1)
            spherical_coordinates[sph_coords] = rgb
    return spherical_coordinates

def sph2cart(phi: float, theta: float, r: float):
    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return (x, y, z)

def cart2sph(cart_coord):
    x,y,z = cart_coord
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z/r)
    phi = math.atan(y/x)
    return (np.rad2deg(theta), np.rad2deg(phi), r)

def fix_phi_based_on_quadrant(x, y, phi):
    quadrant = 0
    if x>0 and y>0:
        quadrant=1
    if x<0 and y>0:
        quadrant=2
    if x<0 and y<0:
        quadrant=3
    if x>0 and y<0:
        quadrant=4
    if quadrant==4:
        phi = 180+phi
    if quadrant==1:
        phi = -90 - (90-phi)
    return phi

def shift_spherical_coordinates(R2):
    x,y,z = sph2cart(0.0, np.pi/2, 1) ## default orientation of the center pixel
    cart_shift = (R2)@np.array([x,y,z])
    theta_m, phi_m, r = cart2sph(cart_shift)
    phi_m = fix_phi_based_on_quadrant(cart_shift[0], cart_shift[1], phi_m)
    return theta_m, -phi_m, r

def get_euler_angles(R2):
    R1 = np.eye(3)
    relative_rotation = R2 @ np.linalg.inv(R1)
    euler_angles = t3d.euler.mat2euler(relative_rotation, axes='sxyz')
    roll, pitch, yaw = euler_angles
    return roll, pitch, yaw

def panorama_stitching(args, qs, qs_ts, cam_imgs, cam_ts, phi_k, theta_k):
    img_qs_ts, predicted_qs, imgs = find_closest_unique_timestamps(qs, qs_ts, cam_imgs, cam_ts)
    output_image = np.zeros((300,500,3))
    ref_q = np.array([1,0,0,0])
    R1 = t3d.quaternions.quat2mat(ref_q)
    i=0
    last_theta1, last_phi1 = None, None
    for q, img in tqdm(zip(predicted_qs[:],imgs[:])):
        # q = [0.96592583, 0., 0.25881905, 0.]
        # i=i+1
        # if i%10!=0:
        #     continue
        
        R2 = t3d.quaternions.quat2mat(q)
        roll, pitch, yaw = get_euler_angles(R2)
        roll, pitch, yaw = np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
        # if abs(roll)>0.2:
        #     continue
        theta1, phi1, r1 = shift_spherical_coordinates(R2)
        theta1 = (theta1-90)*theta_k
        phi1 = phi1*phi_k
        spherical_coordinates = get_spherical_coordinates_for_image(img)
        for coord in spherical_coordinates:
            phi, theta, r = coord
            image_y = theta + theta1
            image_x = phi + phi1
            if image_x<-180:
                image_x = image_x + 360
            if image_x>=180:
                continue
            if image_y<-90:
                continue
            if image_y>=90:
                continue
            image_y = int((image_y+90)*(300/180))
            image_x = int((image_x+180)*(500/360))
            output_image[image_y][image_x] = spherical_coordinates[coord]
    imageio.imwrite(args.basedir + '/' + args.dataset + '/logs/panorama' + str(phi_k) + '_' + str(theta_k)+'.png', output_image.astype(np.uint8))
    return