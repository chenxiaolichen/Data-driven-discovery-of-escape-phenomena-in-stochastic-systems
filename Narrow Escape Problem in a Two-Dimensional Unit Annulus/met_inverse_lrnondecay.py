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


# W*H+b

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)


def net_u(x, y, weights_u, biases_u):
    xy = tf.concat([x, y], 1)
    u = neural_net(xy, weights_u, biases_u)
    u1, u2 = tf.split(u, 2, axis=1)  # 拆分成 u1 和 u2

    return u1, u2

def net_f(x, y, weights_u, biases_u,para1,para2):
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
    f1 = -para1 * (u1 - u2) + D * (u1_xx + u1_yy) + b1 * u1_x + b2 * u1_y + 1
    f2 = -k * (u2 - u1) + D * (u2_xx + u2_yy) + b1 * u2_x + b2 * u2_y + 1

    return f1, f2



def net_drift1(x, para1, para2):
    g1=0.01 *( x + 1.05)
    return g1

def net_drift2(x,y,para1):
    g2=(x +para1 * x ** 3 - y)
    return g2

def u_exact(x,y):
    return (x - x0) * (x1 - x) * (y - y0) * (y1 - y)


##left side
def f_exact(x,y):
    # return -(y - y0) * (y1 - y)-(x - x0) * (x1 - x)+(x - (1 / 3) * x ** 3 - y)*(x1-2*x+x0)*(y-y0)*(y1-y)+0.01*(x+1.05)
    return 0

###true drift
def g1_exact(x):
    return 0.01*(x+1.05)

def g2_exact(x,y):
    return (x - (1 / 3) * x ** 3 - y)

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


layers = [2] + 4* [20] + [2]
L = len(layers)

weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]


para1 = tf.Variable(0.1, dtype=tf.float64)
para2 = tf.Variable(0.1, dtype=tf.float64,trainable=False)
D,w=1,3
n_points = 101  # 每个维度上的点数
total_points = n_points * n_points
r1, r2 = 1 ,2# 内圆半径r1，外圆半径r2
k=0.9
theta = np.linspace(0, 2 * np.pi, n_points)
r_vals = np.linspace(r1, r2, n_points)
# # theta0, theta1,theta2, theta3 = 0,0.1*np.pi, np.pi,np.pi+0.1*np.pi
# # theta0, theta1 = 3 * np.pi / 4,np.pi
theta0, theta1,theta2, theta3 = 0,np.pi / 4, np.pi*3 / 4,np.pi
# # 生成环形区域的网格点
R, Theta = np.meshgrid(r_vals, theta)
X_test = R * np.cos(Theta)
Y_test = R * np.sin(Theta)

r_random = np.random.uniform(r1, r2, total_points)
theta_random = np.random.uniform(0, 2 * np.pi,total_points)
X_random = r_random * np.cos(theta_random)
Y_random = r_random * np.sin(theta_random)

X_test_flat = X_random.flatten()[:, None]
Y_test_flat = Y_random.flatten()[:, None]




xf_tf = tf.placeholder(tf.float64, shape=[None, 1])
yf_tf = tf.placeholder(tf.float64, shape=[None, 1])


f_pred= net_f(xf_tf, yf_tf,weights_u, biases_u, para1,para2)



u1_opt=np.loadtxt('./101-ut1_random.txt')

N_ob=50

u2_opt=np.loadtxt('./101-ut2_random.txt')


choose_idx = np.random.randint(0, total_points, size=N_ob)


x_ob = X_random[choose_idx][:, None]
y_ob = Y_random[choose_idx][:, None]

u1_ob = u1_opt[choose_idx][:, None]
u2_ob = u2_opt[choose_idx][:, None]

u1_ob = tf.convert_to_tensor(u1_ob, dtype=tf.float64)
u2_ob = tf.convert_to_tensor(u2_ob, dtype=tf.float64)

u1_ob_NN, u2_ob_NN = net_u(x_ob, y_ob, weights_u, biases_u)

loss_u1 = tf.reduce_mean(tf.square(u1_ob - u1_ob_NN))  # u1 的观测点损失
loss_u2 = tf.reduce_mean(tf.square(u2_ob - u2_ob_NN))  # u2 的观测点损失

# 总损失
loss_ob = loss_u1 + loss_u2


X_obs = x_ob .flatten()
Y_obs = y_ob .flatten()
plt.figure(figsize=(8, 6))
plt.scatter(X_test_flat, Y_test_flat, color='lightgray', marker='o', s=10, label='Training Points')
plt.scatter(X_obs, Y_obs, color='red', marker='x', s=40, label='Observation Points')
plt.title('Training and Observation Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
#plt.show()

n_boundary_points = 300  # 边界上的点数
n_gate_points = 300

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

##占位符
x_outer_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
y_outer_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
x_inner_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))
y_inner_boundary_ph = tf.placeholder(tf.float64, shape=(None, 1))


x_gate1_tensor  = tf.placeholder(tf.float64, shape=(None, 1))
y_gate1_tensor = tf.placeholder(tf.float64, shape=(None, 1))
x_gate2_tensor = tf.placeholder(tf.float64, shape=(None, 1))
y_gate2_tensor = tf.placeholder(tf.float64, shape=(None, 1))
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

f1_pred, f2_pred = net_f(X_tf, Y_tf, weights_u, biases_u,para1,para2)

# 分别计算 f1 和 f2 的损失
loss_f1 = tf.reduce_mean(tf.square(f1_pred))
loss_f2 = tf.reduce_mean(tf.square(f2_pred))
loss_pde=loss_f1+loss_f2


loss = loss_pde +loss_ob+ (loss_outer_boundary + loss_inner_boundary) + (loss_gate1 + loss_gate2)+(loss_gate12 + loss_gate21)

optimizer_Adam = tf.train.AdamOptimizer(1e-3)
train_op_Adam = optimizer_Adam.minimize(loss)

min_loss = 1e16
saver = tf.train.Saver(max_to_keep=1000)
savedir='./met_inverse_driftpara_weight_50_1e-3_50w_random_ran_duiqi_300dounda'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_record = []
    loss_f_record = []
    loss_ob_record = []
    para1_history=[]
    para2_history=[]
    feed_dict = {
        x_outer_boundary_ph: x_outer_boundary[:, None],
        y_outer_boundary_ph: y_outer_boundary[:, None],
        x_inner_boundary_ph: x_inner_boundary[:, None],
        y_inner_boundary_ph: y_inner_boundary[:, None],
        # X_tf: X_test_flat, Y_tf: Y_test_flat,
        X_tf: X_test_flat, Y_tf: Y_test_flat,
        x_gate1_tensor: x_gate1[:, None], y_gate1_tensor: y_gate1[:, None],
        x_gate2_tensor: x_gate2[:, None], y_gate2_tensor: y_gate2[:, None]
    }
    for i in range(500000+1):
        sess.run(train_op_Adam,feed_dict=feed_dict)
        if i % 1000 == 0:
            temp_loss, temp_loss_pde,temp_loss_ob, temp_loss_outer_boundary, temp_loss_inner_boundary, temp_loss_gate1, temp_loss_gate2, temp_loss_gate12, \
            temp_loss_gate21 = sess.run(
                [loss, loss_pde,loss_ob, loss_outer_boundary, loss_inner_boundary, loss_gate1, loss_gate2, loss_gate12,
                 loss_gate21], feed_dict=feed_dict)
            (para1_train) = sess.run([para1])
            (para2_train) = sess.run([para2])
            para1_history.append(para1_train)
            para2_history.append(para2_train)

            print(f"Iteration {i}, para1: {para1_train},para2: {para2_train}")

            if temp_loss<min_loss:
                min_loss=temp_loss
                u_pred = sess.run(net_u(X_tf, Y_tf, weights_u, biases_u),
                                  feed_dict=feed_dict)

                u1_pred = u_pred[0].reshape((n_points, n_points))
                u2_pred = u_pred[1].reshape((n_points, n_points))

                final_para1 = para1_train
                final_para2 = para2_train

                i_opt=i
                save_path = saver.save(sess, savedir + '/metinver_driftpara_best_model_50_1e-3_50w_random_ran_duiqi_300dounda.ckpt')
                (weights_u_np, biases_u_np) = sess.run([weights_u, biases_u])
                sample_list = {
                    "weights_u": weights_u_np,
                    "biases_u": biases_u_np,
                    "para1": final_para1,
                    "para2": final_para2,
                }

                file_name = './met_inverse_driftpara_weight_50_1e-3_50w_random_ran_duiqi_300dounda/metinver_driftpara_hyper_best_model_50_1e-3_50w_random_ran_duiqi_300dounda.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()

            loss_record.append(temp_loss)
            ut0_u1_pred = u_pred[0].reshape((n_points, n_points))
            ut0_u2_pred = u_pred[1].reshape((n_points, n_points))

            print(
                f"Iteration {i}, Loss: {temp_loss},pde Loss: {temp_loss_pde},loss_ob:{temp_loss_ob},outer_boundary Loss: {temp_loss_outer_boundary},inner_boundary Loss: {temp_loss_inner_boundary},gate1 Loss: {temp_loss_gate1},gate2 Loss: {temp_loss_gate2},gate12 Loss: {temp_loss_gate12},gate21 Loss: {temp_loss_gate21} ")


print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)
print("para1 =", final_para1)
print("para2 =", final_para2)

iterations = list(range(0, len(para1_history) * 1000, 1000))
plt.figure(figsize=(8, 6))
plt.plot(iterations, para1_history, label='Learned $k$ (para1)', color='b')
plt.axhline(y=0.9, color='r', linestyle='--', label='True $k = 0.9$')  # 真值
plt.xlabel('Iteration')
plt.ylabel('$k$ Value')
plt.title('Learning Process of $k$')
plt.legend()
plt.grid(True)
plt.savefig('para1_1_420_1e-3-50ep50w_random_ran_duiqi_300dounda.png')

np.savetxt('para1_record_1e-3-50ep50w_random_ran_duiqi_300dounda.txt',para1_history)


plt.figure(figsize=(8, 6))
plt.plot(iterations, para2_history, label='Learned $k$ (para2)', color='b')
plt.axhline(y=0.9, color='r', linestyle='--', label='True $k = 0.9$')  # 真值
plt.xlabel('Iteration')
plt.ylabel('$k$ Value')
plt.title('Learning Process of $k$')
plt.legend()
plt.grid(True)
