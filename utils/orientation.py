import numpy as np
from scipy.spatial.transform import Rotation as R
from .quaternion import *
from .plots import *
import jax.numpy as jnp
from jax import grad, vmap, jit
from tqdm import tqdm
import os
from jax.numpy.linalg import norm

def acceleration_loss(measured_accl, q):
    gq = jnp.array([0,0,0,-9.8])
    hq = jax_mul(jax_inverse(q), jax_mul(gq, q))
    accl_loss = jnp.array(measured_accl-hq)
    return accl_loss

def calculate_fq(q, tw):
    predicted_q = jax_mul(q, tw)
    # predicted_q = predicted_q/norm(predicted_q)
    return predicted_q

def orientation_loss(qt1, fqt):
    qt1_inv = jax_inverse(qt1)
    logq = jax_log(jax_mul(qt1_inv, fqt))
    return 2*logq

def quaternion_l2_norm_squared(q):
    w,x,y,z = q
    return w**2 + x**2 + y**2 + z**2

def total_loss(measured_accl, qs, taow):
    accl_loss = vmap(acceleration_loss)
    accl_loss_val = vmap(quaternion_l2_norm_squared)(accl_loss(measured_accl, qs))

    fqts = vmap(calculate_fq)
    fqs_0 = calculate_fq(jnp.array([1,0,0,0]), taow[0])
    fqs = fqts(qs[:-1,:], taow[1:])
    fqs = jnp.vstack([fqs_0, fqs])
    qt1s = qs
    or_loss = vmap(orientation_loss)
    or_loss_val = vmap(quaternion_l2_norm_squared)(or_loss(qt1s, fqs))

    loss =  0.5 *jnp.sum(jnp.sqrt(accl_loss_val)) ##+ 0.5 * jnp.sum(jnp.sqrt(or_loss_val)) ##+ 0.5 * jnp.sum(jnp.sqrt(or_loss_val))
    return loss

def get_initial_orientation_trajectory(omegas, ts):
    q0 = np.array([1,0,0,0])
    qs = [q0]
    for i in range(1, len(omegas)):
        qk = qs[i-1]
        wk1 = omegas[i-1]
        tao = 0.5*float(ts[i] - ts[i-1])
        wq = np.array([0, wk1[0]*tao, wk1[1]*tao, wk1[2]*tao])
        qk1 = q_exponential(wq)
        qk1 = q_multiplication(qk, qk1)
        qs.append(qk1)
        # qs.append(q0)
    return np.array(qs).astype(float)

def calculate_taow(omegas, ts):
    taos = [0.5* float(ts[i]-ts[i-1]) for i in range(1,len(ts))]
    taow = np.array([np.concatenate(([0], w*t)) for w,t in zip(omegas[:-1],taos)]).astype(float)
    return taow

def orientation_tracking(args, omegas, accl, ts, gt_ts):
    accl = [np.concatenate(([0], a)) for a in accl][1:]
    qs = get_initial_orientation_trajectory(omegas, ts).astype(float)
    norms = np.linalg.norm(qs, axis=1, keepdims=True)
    qs = qs/norms
    taow = calculate_taow(omegas, ts)
    for i in tqdm(range(args.epochs)):
        loss = total_loss(jnp.array(accl), jnp.array(qs[1:]), jnp.array(taow))
        gradient = grad(total_loss, argnums=1)
        dfx = gradient(jnp.array(accl), jnp.array(qs[1:]), jnp.array(taow)) * (-0.002)
        # hk = dfx - (dfx*qs)*qs
        # nk = np.apply_along_axis(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1, arr=hk)
        # qs = qs*np.cos(np.pi/20) + nk*np.sin(np.pi/20)
        qs[1:] = qs[1:] + dfx
        norms = np.linalg.norm(qs, axis=1, keepdims=True)
        qs = qs/norms
        if i%100==0:
            print('loss :: ', loss)
            log_path = os.path.join(args.basedir, args.dataset)
            log_path = os.path.join(log_path,'logs')
            np.save(log_path + '/q_' + str(i) + '.npy',qs)
            plot_rpy_epoch(log_path, qs, ts,i, gt_ts, args)
    return qs