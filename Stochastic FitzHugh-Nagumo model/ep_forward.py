import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 2):
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


def net_u(x, y, weights, biases):
    xy = tf.concat([x, y], 1)
    u = neural_net(xy, weights, biases)
    return u


def net_f(x, y, weights, biases):
    u = net_u(x, y, weights, biases)
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    f = 0.5* (sigma**2)* (u_xx + u_yy) + (x - (1 / 3) * x ** 3 - y) * u_x + 0.01 * (x + 1.05) * u_y

    return f




layers = [2] + 4 * [32] + [1]
L = len(layers)
weights = [xavier_init([layers[l], layers[l + 1]]) for l in range(L - 1)]
biases = [tf.Variable(tf.zeros((1, layers[l + 1]), dtype=tf.float64)) for l in range(L - 1)]

# 区域边界
x0, x1 = -2, 1
y0, y1 = -3.664125, 2.335875
sigma=1
# 生成训练数据
n_points = 101
x_tf = np.reshape(np.linspace(x0, x1, n_points), [-1, 1])
y_tf = np.reshape(np.linspace(y0, y1, n_points), [-1, 1])
X, Y = np.meshgrid(x_tf, y_tf)
X_flat = X.flatten()[:, None]
Y_flat = Y.flatten()[:, None]

# 边界点
boundary_x = np.concatenate([
    np.full((n_points, 1), x0),  # 左边界
    np.full((n_points, 1), x1),  # 右边界
    np.linspace(x0, x1, n_points).reshape(-1, 1),  # 下边界
    np.linspace(x0, x1, n_points).reshape(-1, 1)  # 上边界
])

boundary_y = np.concatenate([
    np.linspace(y0, y1, n_points).reshape(-1, 1),  # 左边界
    np.linspace(y0, y1, n_points).reshape(-1, 1),  # 右边界
    np.full((n_points, 1), y0),  # 下边界
    np.full((n_points, 1), y1)  # 上边界
])

# 目标值
boundary_target = np.concatenate([
    np.zeros((n_points, 1)),  # 左边界
    np.ones((n_points, 1)),  # 右边界
    np.zeros((n_points, 1)),  # 下边界
    np.zeros((n_points, 1))  # 上边界
])

# 占位符
X_tf = tf.placeholder(tf.float64, shape=[None, 1])
Y_tf = tf.placeholder(tf.float64, shape=[None, 1])
X_boundary = tf.placeholder(tf.float64, shape=[None, 1])
Y_boundary = tf.placeholder(tf.float64, shape=[None, 1])
U_boundary = tf.placeholder(tf.float64, shape=[None, 1])

# PDE 损失
f_pred,range_penalty = net_f(X_tf, Y_tf, weights, biases)
loss_pde = tf.reduce_mean(tf.square(f_pred))

# 边界损失
u_pred_boundary = net_u(X_boundary, Y_boundary, weights, biases)
loss_boundary = tf.reduce_mean(tf.square(u_pred_boundary - U_boundary))

# 总损失
loss = loss_pde + 100*loss_boundary


optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss)

import os
import pickle
saver = tf.train.Saver(max_to_keep=1000)
savedir='./ep_forward_weight'
os.makedirs(savedir)
min_loss=1e-16
# 训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        _, current_loss, pde_loss, boundary_loss = sess.run(
            [train_op, loss, loss_pde, loss_boundary],
            feed_dict={
                X_tf: X_flat,
                Y_tf: Y_flat,
                X_boundary: boundary_x,
                Y_boundary: boundary_y,
                U_boundary: boundary_target
            }
        )

        if i % 1000 == 0:
            print(f"Epoch {i}, Total Loss: {current_loss}, PDE Loss: {pde_loss}, Boundary Loss: {boundary_loss}")
            u_pred_intermediate = sess.run(net_u(X_tf, Y_tf, weights, biases), feed_dict={X_tf: X_flat, Y_tf: Y_flat})
            print("Intermediate u_pred min:", np.min(u_pred_intermediate))
            print("Intermediate u_pred max:", np.max(u_pred_intermediate))

            if current_loss < min_loss:
                min_loss = current_loss
                save_path = saver.save(sess, savedir + '/epforward_best_model.ckpt')
                (weights_u_np, biases_u_np) = sess.run(
                    [weights, biases])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np}

                file_name = './ep_forward_weight/epforward_hyper_best_model.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()

    # 预测
    u_pred = sess.run(net_u(X_tf, Y_tf, weights, biases), feed_dict={X_tf: X_flat, Y_tf: Y_flat})
    u_pred_grid = u_pred.reshape((n_points, n_points))
    np.savetxt('u_pred_grid_EP.txt', u_pred_grid)


