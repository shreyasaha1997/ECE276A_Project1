import pickle
import sys
import time 
import numpy as np
import imageio

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="10"
cfile = "cam/cam" + dataset + ".p"
ifile = "imu/imuRaw" + dataset + ".p"
vfile = "vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
# vicd = read_data(vfile)
toc(ts,"Data import")

print(type(imud))
print(imud.keys())
print(imud['vals'].shape)
vals = imud['vals']
ts = np.squeeze(imud['ts'])
a = vals[:3,:].transpose()
w = vals[3:].transpose()
print(ts.shape)
print(type(w),w.shape, a.shape)
np.save('data/' + dataset + '/omega.npy', w)
np.save('data/' + dataset + '/accl.npy', a)
np.save('data/' + dataset + '/ts.npy', ts)
print('---------')
# print(camd.keys())
# print(camd['cam'].shape)
# print(camd['ts'].shape)
# ts = np.squeeze(camd['ts'])
# images = camd['cam']
# images = np.transpose(images, (3,0,1,2))
# for i,image in enumerate(images):
#   image = image.astype(np.uint8)
#   imageio.imwrite('data/' + dataset + '/imgs/' + str(i) + '.png', image)
# np.save('data/' + dataset + '/images.npy', images)
# np.save('data/' + dataset + '/cam_ts.npy', ts)
# print('---------')
# print(type(vicd))
# print(vicd.keys())
# print(vicd['rots'].shape, vicd['ts'].shape)
# ts = np.squeeze(vicd['ts'])
# rots = vicd['rots']
# rots = np.transpose(rots, (2,0,1))
# np.save('data/' + dataset + '/vicd_ts.npy', ts)
# np.save('data/' + dataset + '/vicd_rots.npy', rots)
# print('---------')





