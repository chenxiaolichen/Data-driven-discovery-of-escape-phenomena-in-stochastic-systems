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
from tensorflow.keras.callbacks import TensorBoard

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
    in_dim, out_dim = size
    lecun_stddev = np.sqrt(1.0 / in_dim)
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=lecun_stddev, dtype=tf.float64), dtype=tf.float64)

def net_u(x, y, weights_u, biases_u):
    xy = tf.concat([x, y], 1)
    u = neural_net(xy, weights_u, biases_u)
    u1, u2 = tf.split(u, 2, axis=1)  # 拆分成 u1 和 u2


    return u1, u2


def net_f(x, y, weights_u, biases_u):
    u1, u2 = net_u(x, y, weights_u, biases_u)

    u1_x = tf.gradients(u1, x)[0]
    u1_y = tf.gradients(u1, y)[0]
    u1_xx = tf.gradients(u1_x, x)[0]
    u1_yy = tf.gradients(u1_y, y)[0]

    u2_x = tf.gradients(u2, x)[0]
    u2_y = tf.gradients(u2, y)[0]
    u2_xx = tf.gradients(u2_x, x)[0]
    u2_yy = tf.gradients(u2_y, y)[0]

    # 漂移项 b1 和 b2
    b1 = -y * ((-w * r1**2 * (x**2 + y**2) + w * r1**2 * r2**2) /
               (((x**2 + y**2)**0.5) * (r2**2 - r1**2)))
    b2 = x * ((-w * r1**2 * (x**2 + y**2) + w * r1**2 * r2**2) /
              (((x**2 + y**2)**0.5) * (r2**2 - r1**2)))

    # 方程残差 f1 和 f2
    f1 = -k * (u1 - u2) + D * (u1_xx + u1_yy) + b1 * u1_x + b2 * u1_y + 1
    f2 = -k * (u2 - u1) + D * (u2_xx + u2_yy) + b1 * u2_x + b2 * u2_y + 1

    return f1, f2

def compute_boundary_loss1(x_boundary_ph, y_boundary_ph, weights, biases, radius):

    u1, u2 = net_u(x_boundary_ph, y_boundary_ph, weights, biases)

    u1_x = tf.gradients(u1, x_boundary_ph)[0]
    u1_y = tf.gradients(u1, y_boundary_ph)[0]
    u2_x = tf.gradients(u2, x_boundary_ph)[0]
    u2_y = tf.gradients(u2, y_boundary_ph)[0]


    nx = x_boundary_ph / radius
    ny = y_boundary_ph / radius
    u1_r = u1_x * nx + u1_y * ny
    u2_r = u2_x * nx + u2_y * ny


    loss_u1 = tf.reduce_mean(tf.square(u1_r))
    loss_u2 = tf.reduce_mean(tf.square(u2_r))

    return loss_u1 + loss_u2

def compute_boundary_loss2(x_boundary_ph, y_boundary_ph, weights, biases, radius):

    u1, u2 = net_u(x_boundary_ph, y_boundary_ph, weights, biases)


    loss_u1 = tf.reduce_mean(tf.square(u1))
    loss_u2 = tf.reduce_mean(tf.square(u2))

    return loss_u1 + loss_u2

def boundary_loss(u_pred, x_boundary_ph, y_boundary_ph, radius):

    u_pred_x = tf.gradients(u_pred, x_boundary_ph)[0]
    u_pred_y = tf.gradients(u_pred, y_boundary_ph)[0]

    nx = x_boundary_ph / radius
    ny = y_boundary_ph / radius


    u_pred_r = u_pred_x * nx + u_pred_y * ny


    loss = tf.reduce_mean(tf.square(u_pred_r))

    return loss

layers = [2] + 4 * [20] + [2]
L = len(layers)
weights_u = [xavier_init([layers[l], layers[l + 1]]) for l in range(L - 1)]
biases_u = [tf.Variable(tf.zeros((1, layers[l + 1]), dtype=tf.float64)) for l in range(L - 1)]


D,w=1,3
n_points = 101
total_points = n_points * n_points
r1, r2 = 1,2# 内圆半径r1，外圆半径r2
k=0.9
theta = np.linspace(0, 2 * np.pi, n_points)
r_vals = np.linspace(r1, r2, n_points)

theta0, theta1,theta2, theta3 = 0,np.pi / 4, np.pi*3 / 4,np.pi
# # 生成环形区域的网格点
R, Theta = np.meshgrid(r_vals, theta)
X_test = R * np.cos(Theta)
Y_test = R * np.sin(Theta)

r_random = np.random.uniform(r1, r2, total_points)
theta_random = np.random.uniform(0, 2 * np.pi,total_points)
X_random = r_random * np.cos(theta_random)
Y_random = r_random * np.sin(theta_random)
X_random_flat = X_random[:, None]
Y_random_flat = Y_random[:, None]

n_boundary_points = 200  # 边界上的点数
n_gate_points = 100

# 内边界（半径 r1）上的点，去除吸收门的部分
theta_inner = np.linspace(0, 2 * np.pi, n_boundary_points)
x_inner_boundary = r1 * np.cos(theta_inner)
y_inner_boundary = r1 * np.sin(theta_inner)


# 外边界（半径 r1）上的点
theta_outer = np.linspace(0, 2 * np.pi, n_boundary_points)
mask_inner = np.ones(n_boundary_points, dtype=bool)
# 去除吸收门1的部分
mask_inner &= ~np.logical_and(theta_inner >= theta0, theta_inner <= theta1)
# 去除吸收门2的部分
mask_inner &= ~np.logical_and(theta_inner >= theta2, theta_inner <= theta3)
x_outer_boundary = r2 * np.cos(theta_outer[mask_inner])
y_outer_boundary = r2 * np.sin(theta_outer[mask_inner])

# 吸收门1的角度范围：θ0 到 θ1
theta_gate1 = np.linspace(theta0, theta1, n_gate_points)
x_gate1 = r2 * np.cos(theta_gate1)  # 吸收门1 θ0 到 θ1 之间的 x 坐标
y_gate1 = r2 * np.sin(theta_gate1)  # 吸收门1 θ0 到 θ1 之间的 y 坐标

# # 吸收门2的角度范围：θ2 到 θ3
theta_gate2 = np.linspace(theta2, theta3, n_gate_points)
x_gate2 = r2 * np.cos(theta_gate2)  # 吸收门2 θ2 到 θ3 之间的 x 坐标
y_gate2 = r2 * np.sin(theta_gate2)  # 吸收门2 θ2 到 θ3 之间的 y 坐标


x_outer_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
y_outer_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
x_inner_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
y_inner_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))


x_gate1_tensor  = tf.placeholder(tf.float64, shape=(None, 1))
y_gate1_tensor = tf.placeholder(tf.float64, shape=(None, 1))
x_gate2_tensor = tf.placeholder(tf.float64, shape=(None, 1))
y_gate2_tensor = tf.placeholder(tf.float64, shape=(None, 1))


plt.figure(figsize=(8, 8))
plt.scatter(X_random_flat, Y_random_flat, s=1, c='blue', label='Training Points')
plt.plot(x_inner_boundary, y_inner_boundary, 'g.', label='Inner Boundary')
plt.plot(x_outer_boundary, y_outer_boundary, 'r.', label='Outer Boundary')
plt.plot(x_gate1, y_gate1, 'm.', linewidth=2, label='Gate 1')
plt.plot(x_gate2, y_gate2, 'y.', linewidth=2, label='Gate 2')
plt.legend()
plt.axis('equal')
plt.title('Training Points and Boundary Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# 计算内边界损失
loss_inner_boundary = compute_boundary_loss2(
    x_inner_boundary_ph, y_inner_boundary_ph, weights_u, biases_u, r1)

# 计算外边界损失
loss_outer_boundary = compute_boundary_loss1(
    x_outer_boundary_ph, y_outer_boundary_ph, weights_u, biases_u, r2)


X_tf = tf.placeholder(tf.float64, shape=[None, 1])
Y_tf = tf.placeholder(tf.float64, shape=[None, 1])
# 吸收门1上的解 u1 和 u2
u1_pred_gate1, u2_pred_gate1 = net_u(x_gate1_tensor, y_gate1_tensor, weights_u, biases_u)

# 吸收门2上的解 u1 和 u2
u1_pred_gate2, u2_pred_gate2 = net_u(x_gate2_tensor, y_gate2_tensor, weights_u, biases_u)

loss_gate1 = tf.reduce_mean(tf.square(u1_pred_gate1))  # 吸收门1的损失
loss_gate2 = tf.reduce_mean(tf.square(u2_pred_gate2))  # 吸收门2的损失

# 计算吸收门1处的损失
loss_gate12 = boundary_loss(u2_pred_gate1, x_gate1_tensor, y_gate1_tensor, r2)

# 计算吸收门2处的损失
loss_gate21 = boundary_loss(u1_pred_gate2, x_gate2_tensor, y_gate2_tensor, r2)

# 计算 f 的预测
f1_pred, f2_pred = net_f(X_tf, Y_tf, weights_u, biases_u)

# 分别计算 f1 和 f2 的损失
loss_f1 = tf.reduce_mean(tf.square(f1_pred))
loss_f2 = tf.reduce_mean(tf.square(f2_pred))

# 总损失
loss_pde = loss_f1 + loss_f2
min_loss = 1e16

loss = loss_pde + (loss_outer_boundary + loss_inner_boundary) + (loss_gate1 + loss_gate2)+(loss_gate12 + loss_gate21)

optimizer = tf.train.AdamOptimizer(5e-3)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver(max_to_keep=1000)
savedir='./met_forward_weight_random'
loss_pde_history = []
loss_outer_boundary_history = []
loss_inner_boundary_history = []
loss_gate1_history = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = {
        x_outer_boundary_ph: x_outer_boundary[:, None],
        y_outer_boundary_ph: y_outer_boundary[:, None],
        x_inner_boundary_ph: x_inner_boundary[:, None],
        y_inner_boundary_ph: y_inner_boundary[:, None],
        # X_tf: X_test_flat, Y_tf: Y_test_flat,
    X_tf: X_random_flat, Y_tf: Y_random_flat,
        x_gate1_tensor:x_gate1[:, None], y_gate1_tensor:y_gate1[:, None] ,
    x_gate2_tensor:x_gate2[:, None],  y_gate2_tensor:y_gate2[:, None]
    }
    loss_history = []
    for i in range(70000):
        sess.run(train_op, feed_dict=feed_dict)
        if i % 1000 == 0:

            temp_loss, temp_loss_pde,temp_loss_outer_boundary ,temp_loss_inner_boundary,temp_loss_gate1 , temp_loss_gate2,temp_loss_gate12 , \
            temp_loss_gate21= sess.run([loss, loss_pde,loss_outer_boundary ,loss_inner_boundary,loss_gate1 , loss_gate2,loss_gate12 , loss_gate21], feed_dict=feed_dict)
            u_pred_intermediate = sess.run(net_u(X_tf, Y_tf, weights_u, biases_u), feed_dict=feed_dict)
            u1 = u_pred_intermediate[0]  # 提取 u1
            u2 = u_pred_intermediate[ 1]  # 提取 u2

            # 计算 u1 和 u2 的最小值和最大值
            u1_min = np.min(u1)
            u1_max = np.max(u1)
            u2_min = np.min(u2)
            u2_max = np.max(u2)

            # 输出结果
            print("u1 min:", u1_min)
            print("u1 max:", u1_max)
            print("u2 min:", u2_min)
            print("u2 max:", u2_max)

            current_loss = sess.run(loss, feed_dict=feed_dict)
            loss_history.append(current_loss)
            print(f"Epoch {i}, Loss: {current_loss}")
            loss_pde_history.append(temp_loss_pde)
            loss_outer_boundary_history.append(temp_loss_outer_boundary)
            loss_inner_boundary_history.append(temp_loss_inner_boundary)
            loss_gate1_history.append(temp_loss_gate1)

            if temp_loss < min_loss:
                min_loss = temp_loss
                u_pred = sess.run(net_u(X_tf, Y_tf, weights_u, biases_u),
                                  feed_dict=feed_dict)
                u1_pred = u_pred[0]
                u2_pred = u_pred[1]

                save_path = saver.save(sess, savedir + '/metforward_best_model_random.ckpt')
                (weights_u_np, biases_u_np) = sess.run(
                    [weights_u, biases_u])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np}

                file_name = './met_forward_weight_random/metforward_hyper_best_model_random.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)


            ut0_u1_pred = u_pred[0]
            ut0_u2_pred= u_pred[1]

            print(f"Iteration {i}, Loss: {temp_loss},pde Loss: {temp_loss_pde},outer_boundary Loss: {temp_loss_outer_boundary},inner_boundary Loss: {temp_loss_inner_boundary},gate1 Loss: {temp_loss_gate1},gate2 Loss: {temp_loss_gate2},gate12 Loss: {temp_loss_gate12},gate21 Loss: {temp_loss_gate21} ")


    f1_vals, f2_vals = sess.run([f1_pred, f2_pred], feed_dict=feed_dict)
    f1_vals = np.squeeze(f1_vals)
    f2_vals = np.squeeze(f2_vals)

    plt.figure()
    plt.scatter(X_test, Y_test, c=f1_vals, cmap='coolwarm', s=5)
    plt.colorbar(label='f1_pred')
    plt.title("PDE Residuals")
    plt.xlabel('x')
    plt.ylabel('y')
   # plt.show()
    plt.savefig('201_f1_random.png')

    plt.figure()
    plt.scatter(X_test, Y_test, c=f2_vals, cmap='coolwarm', s=5)
    plt.colorbar(label='f2_pred')
    plt.title("PDE Residuals")
    plt.xlabel('x')
    plt.ylabel('y')
   # plt.show()
    plt.savefig('201_f2_random.png')

np.savetxt('101-ut1_random.txt', ut0_u1_pred)
np.savetxt('101-ut2_random.txt', ut0_u2_pred)

