#import tensorflow as tf
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
import time


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y
#W*H+b

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)




def net_u(xu, weights, biases):
    u = (xu-x0) * (x1-xu) *neural_net(xu, weights, biases)
    return u


def net_f(xf, weights,biases):
    u =(xf-x0) * (x1-xf) * neural_net(xf, weights, biases)
    u_x = tf.gradients(u, xf)[0]
    u_xx = tf.gradients(u_x, xf)[0]
    return (xf-xf**3)*u_x+0.5*u_xx

def f_exact(x):
    return -1



layers = [1] + 4 * [20] + [1]
L = len(layers)

weights = [xavier_init([layers[l], layers[l + 1]]) for l in range(0, L - 1)]
biases = [tf.Variable(tf.zeros((1, layers[l + 1]), dtype=tf.float64)) for l in range(0, L - 1)]
x0, x1 = -1,1
xf = np.reshape(np.linspace(x0, x1, 101), [-1, 1])
xf_tf = tf.to_double(xf)

f_target = f_exact(xf)

f_pred = net_f(xf_tf,weights, biases)
loss = tf.reduce_mean(tf.square(f_target - f_pred))
optimizer_Adam = tf.train.AdamOptimizer(1e-3)
train_op_Adam = optimizer_Adam.minimize(loss)

x_test = np.reshape(np.linspace(x0, x1, 101), [-1, 1])
u_test = u_exact(x_test)
min_loss = 1e16

saver = tf.train.Saver(max_to_keep=1000)
savedir='./met_forward_weight'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    u_pred = net_u(x_test, weights, biases)

    loss_record = []
    for i in range(20000):
        sess.run(train_op_Adam)
        if i % 1000 == 0:
            temp_loss = sess.run(loss)
            if temp_loss < min_loss:
                min_loss = temp_loss
                ut_opt = np.reshape(sess.run(u_pred), [-1, 1])
                error_u_opt = np.linalg.norm(ut_opt - u_test, 2) / np.linalg.norm(u_test, 2)
                i_opt = i
                save_path = saver.save(sess, savedir + '/metforward_best_model.ckpt')
                (weights_u_np, biases_u_np) = sess.run(
                    [weights, biases])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np}

                file_name = './met_forward_weight/metforward_hyper_best_model.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)


            loss_record.append(temp_loss)
            ut0 = np.reshape(sess.run(u_pred), [-1, 1])
            # print('  %d    %8.2e  %8.2e' % (i, temp_loss, error_u))

# np.savetxt('101-ut0-met.txt', ut0)
print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)

# plt.plot(x_0,t_need,'k',)
# plt.plot(x_0,ut0,'r--')
# plt.xlabel('x', fontsize=12)
# plt.ylabel('MET', fontsize=12)
# plt.ylim(-0.05, 2.5)
# plt.legend(['MC','PINN'] ,frameon=False)
# plt.title(r'$\mathrm{\sigma}=1$', fontsize=13)
# plt.savefig('1d_met_forward_sigma1.png')
