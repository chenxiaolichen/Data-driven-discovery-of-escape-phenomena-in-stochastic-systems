import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.special as sci
np.random.seed(1234)
tf.set_random_seed(1234)
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle

dt = 0.0005
n = 101
a = 1
sigma = 1
rng = np.random.default_rng(1234)
x_0 = np.linspace(-1, 1, 101)
def drift(x):
    return x - x**3
def diffusivity(x):
    dW = rng.normal(loc=0, scale=np.sqrt(dt), size=x.shape)
    return sigma*dW
t_end = np.zeros((101, 1000))
for i in range(n):
    for j in range(1000):
        t = 0
        x = x_0[i]
        while -a < x < a:
            x += dt * drift(x) + diffusivity(x)
            t += dt
        t_end[i][j] = t

    #print(f"Initial position {i} completed")

t_need = np.mean(t_end, axis=1)
np.savez('sigma1met_MC_1000_101.npz', t_need=t_need, x_0=x_0)
