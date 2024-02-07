import numpy as np
from scipy.spatial.transform import Rotation as R
import jax.numpy as jnp
from jax import grad, vmap
from jax.numpy.linalg import norm

def q_exponential(q):
    norm = np.linalg.norm(q[1:])
    exp_scalar = np.exp(q[0]) * np.cos(norm)
    exp_vector = np.exp(q[0]) * (q[1:] / norm) * np.sin(norm)
    exp_q = np.array([exp_scalar, *exp_vector])
    return exp_q

def q_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    q_mul = np.array([w, x, y, z])
    return q_mul

def q_inverse(q):
    norm_squared = np.sum(q**2)
    q = np.array([q[0] / norm_squared, -q[1] / norm_squared, -q[2] / norm_squared, -q[3] / norm_squared])    
    return q

def jax_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    A = jnp.array([w, x, y, z])
    return A

def jax_inverse(q):
    w, x, y, z = q
    norm_sq = w**2 + x**2 + y**2 + z**2
    inverse = jnp.array([w, -x, -y, -z]) / norm_sq
    return inverse

def jax_exp(q):
    w, v = q[0], q[1:]
    norm_v = jnp.linalg.norm(v)
    exp_w = jnp.exp(w)
    v_normalized = v / norm_v
    A = jnp.array([jnp.cos(norm_v), *jnp.sin(norm_v) / norm_v * v_normalized]) * exp_w
    return A

def jax_log(q):
    norm_q = jnp.linalg.norm(q)
    v = q[1:]
    qs = q[0]
    norm_v = jnp.linalg.norm(v)
    log_q = jnp.array([jnp.log(norm_q), *v*(jnp.arccos(qs/norm_q)/norm_v)])
    return log_q