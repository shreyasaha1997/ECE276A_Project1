import numpy as np
import configargparse
import os
from utils.dataloader import *
from utils.orientation import *
from utils.panorama import *
from utils.plots import *
import matplotlib.pyplot as plt 

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--orientation_tracking", action='store_true', 
                        help='calculate orientation')
    parser.add_argument("--panaroma_creation", action='store_true', 
                        help='create panaroma')
    parser.add_argument("--basedir", type=str, default='data/', 
                        help='where to load the data from')
    parser.add_argument("--dataset", type=str, default='11', 
                        help='experiment name')
    parser.add_argument("--epochs", type=int, default='600', 
                        help='number of epochs')
    parser.add_argument("--panorama_qs", type=int, default='500', 
                        help='get estimated qs for panorama')
    return parser


if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    accls, omegas, gt_rots, ts, gt_ts, cam_imgs, cam_ts = load_data(args)
    if args.orientation_tracking:
        qs = orientation_tracking(args, omegas, accls, ts, gt_ts)
        # plot_rpy(args, qs, ts, gt_ts)
    if args.panaroma_creation:
        panorama_qs = np.load(os.path.join(os.path.join(os.path.join(args.basedir,args.dataset), 'logs'), 'q_' + str(args.panorama_qs)+'.npy'))
        panorama_stitching(args, panorama_qs, ts, cam_imgs, cam_ts,1,1)
        panorama_stitching(args, panorama_qs, ts, cam_imgs, cam_ts,0.01,1)
        panorama_stitching(args, panorama_qs, ts, cam_imgs, cam_ts,1,0.01)
    print("done")


