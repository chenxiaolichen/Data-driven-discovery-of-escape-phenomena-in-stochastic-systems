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

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)

    return Y


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)


def net_u(x, y, weights_u, biases_u):
    xy = tf.concat([x, y], 1)
    u = (x - x0) * (x1 - x) * (y - y0) * (y1 - y)*neural_net(xy, weights_u, biases_u)
    return u

def net_f(x, y, weights_u, biases_u):

    u = (x - x0) * (x1 - x) * (y - y0) * (y1 - y)*neural_net(tf.concat([x, y], 1), weights_u, biases_u)
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    f =0.5 * (sigma**2)*(u_xx + u_yy) + (x - (1 / 3) * x ** 3 - y) * u_x + 0.01 * (x + 1.05) * u_y

    return f

def f_exact(x,y):
    # return -(y - y0) * (y1 - y)-(x - x0) * (x1 - x)+(x - (1 / 3) * x ** 3 - y)*(x1-2*x+x0)*(y-y0)*(y1-y)+0.01*(x+1.05)
    return -1

def u_exact(x, y):
    u_exact = (x - x0) * (x1 - x) * (y - y0) * (y1 - y)
    return u_exact


layers = [2] + 4 * [20] + [1]
L = len(layers)
weights_u = [xavier_init([layers[l], layers[l + 1]]) for l in range(L - 1)]
biases_u = [tf.Variable(tf.zeros((1, layers[l + 1]), dtype=tf.float64)) for l in range(L - 1)]
x0, x1 = -2, 1
y0, y1 = -3.664125, 2.335875
n_points = 101
sigma=1

x_tf = np.reshape(np.linspace(x0, x1, n_points), [-1, 1])
y_tf = np.reshape(np.linspace(y0, y1, n_points), [-1, 1])
X_test, Y_test = np.meshgrid(x_tf, y_tf)
X_test_flat = X_test.flatten()[:, None]
Y_test_flat = Y_test.flatten()[:, None]


u_test = u_exact(X_test_flat, Y_test_flat).reshape((n_points, n_points))

X_tf = tf.placeholder(tf.float64, shape=[None, 1])
Y_tf = tf.placeholder(tf.float64, shape=[None, 1])
# XY_tf = tf.placeholder(tf.float64, shape=[None, 2])

##boundary
# x_1 = np.concatenate([np.full((n_points, 1), -2), np.linspace(y0, y1, n_points).reshape(-1, 1)], axis=1)
# y_1= np.concatenate([np.linspace(x0, x1, n_points).reshape(-1, 1),np.full((n_points, 1), -3.664125)], axis=1)
# x_2= np.concatenate([np.full((n_points, 1), 1), np.linspace(y0, y1, n_points).reshape(-1, 1)], axis=1)
# y_2= np.concatenate([np.linspace(x0, x1, n_points).reshape(-1, 1),np.full((n_points, 1),2.335875)], axis=1)
#
# boundary_all = np.vstack([x_1, y_1, x_2, y_2])
# boundary_x=boundary_all[:, 0:1]
# boundary_y=boundary_all[:, 1:2]
#
# X_boundary = tf.placeholder(tf.float64, shape=[None, 1])
# Y_boundary = tf.placeholder(tf.float64, shape=[None, 1])


f_pred = net_f(X_tf, Y_tf, weights_u, biases_u)
f_target = f_exact(X_test_flat, Y_test_flat)
min_loss = 1e16


# u_pred_boundary = net_u(X_boundary, Y_boundary, weights_u, biases_u)
# u_targe_boundary = u_exact(boundary_x, boundary_y)
# loss_boundary = tf.reduce_mean(tf.square(u_pred_boundary - u_targe_boundary))
loss_pde=tf.reduce_mean(tf.square(f_pred-f_target))
# loss = loss_pde +100*loss_boundary
loss = loss_pde
optimizer = tf.train.AdamOptimizer(5e-3)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver(max_to_keep=1000)
savedir='./met_forward_weight'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(100000):
        sess.run(train_op, feed_dict={X_tf: X_test_flat, Y_tf:Y_test_flat})
        if i % 1000 == 0:


            temp_loss, temp_loss_pde= sess.run([loss, loss_pde], feed_dict={X_tf: X_test_flat, Y_tf: Y_test_flat})


            if temp_loss < min_loss:
                min_loss = temp_loss
                u_pred = sess.run(net_u(X_tf, Y_tf, weights_u, biases_u),
                                  feed_dict={X_tf: X_test_flat, Y_tf: Y_test_flat})
                u_opt_grid = u_pred.reshape((n_points, n_points))

                error_u_opt = np.linalg.norm(u_opt_grid - u_test, 2) / np.linalg.norm(u_test, 2)
                save_path = saver.save(sess, savedir + '/metforward_best_model.ckpt')
                (weights_u_np, biases_u_np) = sess.run(
                    [weights_u, biases_u])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np}

                file_name = './met_forward_weight/metforward_hyper_best_model.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()
                i_opt=i

            ut0 = u_pred.reshape((n_points, n_points))
            error_u = np.linalg.norm(ut0 - u_test, 2) / np.linalg.norm(u_test, 2)
            # print(f"Iteration {i}, Loss: {temp_loss},pde Loss: {temp_loss_pde} Boundary Loss: {temp_loss_boundary},error_u:{error_u}")
            #print(f"Iteration {i}, Loss: {temp_loss},pde Loss: {temp_loss_pde} ,error_u:{error_u}")
            # error_u = np.linalg.norm(ut_opt - u_test, 2) / np.linalg.norm(u_test, 2)
            # print('  %8.2e '% (error_u))
print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)
np.savetxt('101-ut0-met.txt', ut0)
# x_vals = np.linspace(x0, x1, n_points)
# y_vals = np.linspace(y0, y1, n_points)
# u_opt_grid =ut0.reshape((n_points, n_points))
#
# plt.figure()
# plt.pcolormesh(x_vals, y_vals, ut0,  cmap='viridis')
# plt.colorbar()
# plt.xlabel('x', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.title(r'$\mathrm{\sigma}=1$', fontsize=13)
# plt.savefig('2d_met_forward_sigma1.png')


