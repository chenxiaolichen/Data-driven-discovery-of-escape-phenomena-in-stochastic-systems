0#import tensorflow as tf
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


def net_u(x,y, weights, biases):
    xy = tf.concat([x, y], 1)

    u=neural_net(xy, weights, biases)
    return u


def net_f(x, y, weights_u, biases_u, para2):
    xy = tf.concat([x, y], 1)
    u =neural_net(xy, weights_u, biases_u)
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    g1=para2*( x + 1.05)
    g2=(x- (1 / 3)* x ** 3 - y)
    return 0.5 * (u_xx + u_yy) + g2 * u_x + g1 * u_y


def net_drift1(x, para2):
    g1=para2 *( x +1.05)
    return g1

# def net_drift2(x,y,para2):
#     g2=para2 *( x +1.05)
#     #g2=(x +para2 * x ** 3 - y)
#     return g2

def u_exact(x,y):
    return (x - x0) * (x1 - x) * (y - y0) * (y1 - y)


##left side
def f_exact(x,y):
    # return -(y - y0) * (y1 - y)-(x - x0) * (x1 - x)+(x - (1 / 3) * x ** 3 - y)*(x1-2*x+x0)*(y-y0)*(y1-y)+0.01*(x+1.05)
    return 0

###true drift
def g1_exact(x):
    return 0.01*(x+1.05)

# def g2_exact(x,y):
#     #return (x - (1 / 3) * x ** 3 - y)
#     return 0.01*(x+1.05)



layers = [2] + 4* [32] + [1]
L = len(layers)

weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]


para2=tf.Variable(0.1,dtype=tf.float64)
x0, x1 = -2, 1
y0, y1 = -3.664125, 2.335875
n_points=101

x_tf = np.reshape(np.linspace(x0, x1, n_points), [-1, 1])
y_tf = np.reshape(np.linspace(y0, y1, n_points), [-1, 1])
X_test, Y_test = np.meshgrid(x_tf, y_tf)
X_test_flat = X_test.flatten()[:, None]
Y_test_flat = Y_test.flatten()[:, None]

xf_tf = tf.placeholder(tf.float64, shape=[None, 1])
yf_tf = tf.placeholder(tf.float64, shape=[None, 1])


f_target = f_exact(x_tf,yf_tf)
f_pred= net_f(xf_tf, yf_tf,weights_u, biases_u,para2)

u_opt=np.loadtxt('./u_pred_grid_EP.txt')
N_ob=50

choose_ob = np.random.choice(X_test.size, size=N_ob, replace=False)


X_obs = X_test.flatten()[choose_ob][:, None]
Y_obs = Y_test.flatten()[choose_ob][:, None]

u_ob = u_opt.flatten()[choose_ob][:, None]

for i in range(5):
    print(f"Obs {i}: (X, Y) = ({X_obs[i][0]}, {Y_obs[i][0]}), u = {u_ob[i][0]}")

u_ob_NN = net_u(X_obs, Y_obs, weights_u, biases_u)
loss_ob=tf.reduce_mean(tf.square(u_ob-u_ob_NN))

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
X_boundary = tf.placeholder(tf.float64, shape=[None, 1])
Y_boundary = tf.placeholder(tf.float64, shape=[None, 1])
U_boundary = tf.placeholder(tf.float64, shape=[None, 1])


# 边界损失
u_pred_boundary = net_u(X_boundary, Y_boundary, weights_u, biases_u)
loss_boundary = tf.reduce_mean(tf.square(u_pred_boundary - U_boundary))

plt.figure(figsize=(8, 6))
plt.scatter(X_test, Y_test, color='lightgray', marker='o', s=10, label='Training Points')
plt.scatter(X_obs, Y_obs, color='red', marker='x', s=40, label='Observation Points')
plt.title('Training and Observation Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True)
# plt.show()
loss_f=tf.reduce_mean(tf.square(f_pred-f_target))
u_test = u_exact(X_test_flat, Y_test_flat)
drift1_test = g1_exact(X_test_flat).reshape(n_points, n_points)

#drift2_test=g2_exact(X_test_flat,Y_test_flat).reshape(n_points, n_points)

# loss_g2=tf.reduce_mean(tf.square(drift2_test-drift2_pred))
# loss_g1=tf.reduce_mean(tf.square(drift1_test-drift1_pred))
loss =loss_f+loss_ob+100*loss_boundary

# optimizer_Adam = tf.train.AdamOptimizer(5e-3)
# train_op_Adam = optimizer_Adam.minimize(loss)

initial_learning_rate = 1e-3
global_step = tf.Variable(0, trainable=False)
decay_steps =1000
decay_rate = 0.96
staircase = True

learning_rate=tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,decay_rate,staircase=staircase)

optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
train_op_Adam = optimizer_Adam.minimize(loss, global_step=global_step)


min_loss = 1e16
saver = tf.train.Saver(max_to_keep=1000)
savedir='./ep_inverse_driftpara_weight_30w_50_1e-3_decay1000'

para1_record = []
para2_record = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss_record = []
    loss_f_record = []
    loss_ob_record = []
    for i in range(300000+1):
        sess.run(train_op_Adam,feed_dict={
                xf_tf: X_test_flat, yf_tf: Y_test_flat,
                X_boundary: boundary_x,
                Y_boundary: boundary_y,
                U_boundary: boundary_target
            })
        if i % 1000 == 0:
            temp_loss, loss_pde_result, loss_ob_result = sess.run([loss,
                    loss_f, loss_ob], feed_dict={xf_tf: X_test_flat, yf_tf: Y_test_flat, X_boundary: boundary_x,
                           Y_boundary: boundary_y,
                           U_boundary: boundary_target})
            (para2_train) = sess.run([para2])

            print(f"Iteration {i}, para1: {para2_train}")

            para2_record.append(para2_train)
            if temp_loss<min_loss:
                min_loss=temp_loss
                u_pred =  sess.run(net_u(xf_tf, yf_tf, weights_u, biases_u),feed_dict={xf_tf: X_test_flat, yf_tf: Y_test_flat})
                drift1_pred = sess.run(net_drift1(xf_tf, para2),feed_dict={xf_tf: X_test_flat})
                #drift2_pred = sess.run(net_drift2(xf_tf, yf_tf, para2), feed_dict={xf_tf: X_test_flat, yf_tf: Y_test_flat})
                f_p = sess.run( net_f(tf.cast(xf_tf, tf.float64), tf.cast(yf_tf, tf.float64), weights_u, biases_u,para2), feed_dict={xf_tf: X_test_flat, yf_tf: Y_test_flat})

                ut_opt =u_pred.reshape(n_points, n_points)
                ft_opt = f_p.reshape(n_points, n_points)
                drift1_opt = drift1_pred.reshape(n_points, n_points)
                #drift2_opt = drift2_pred.reshape(n_points, n_points)


                final_para2 = para2_train
                u_ob_NN_opt=np.reshape(sess.run(u_ob_NN),[-1,1])
                i_opt=i
                save_path = saver.save(sess, savedir + '/epinver_driftpara_best_model_30w_50_1e-3_decay1000.ckpt')
                (weights_u_np, biases_u_np) = sess.run([weights_u, biases_u])
                sample_list = {
                    "weights_u": weights_u_np,
                    "biases_u": biases_u_np,

                    "para2": final_para2,
                }

                file_name = './ep_inverse_driftpara_weight_30w_50_1e-3_decay1000/epinver_driftpara_hyper_best_model_30w_50_1e-3_decay1000.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()
            loss_record.append(temp_loss)
            ut0 = u_pred.reshape((n_points, n_points))
            drift1_result = drift1_pred.reshape(n_points, n_points)
            #drift2_result = drift2_pred.reshape(n_points, n_points)
            error_drift1 = np.linalg.norm(drift1_result - drift1_test, 2) / np.linalg.norm(drift1_test, 2)
            #error_drift2 = np.linalg.norm(drift2_result - drift2_test, 2) / np.linalg.norm(drift2_test, 2)
            print(
                f"Iteration {i}, Loss: {temp_loss},pde Loss: {loss_pde_result} ob Loss: {loss_ob_result},error_drift1:{error_drift1},drift2:{para2_train}")

            # print(
            #     f"Iteration {i}, Loss: {temp_loss},pde Loss: {loss_pde_result} ob Loss: {loss_ob_result}")
    print("Final Parameters:")

    print("para2 =", final_para2)
print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)
# plt.figure()
# plt.plot(range(0, len(para1_record) * 1000, 1000), para1_record, label="para1", color='b')
# plt.axhline(y=1.05, color='r', linestyle='--', label='True $para = 1.05$')  # 真值
# plt.title("Training of para")
# plt.xlabel("Iterations")
# plt.ylabel("parameters value")
#
# plt.savefig('para1.05_30w_50_5e-3_decay2000.png')
#
# np.savetxt('para1.05_record_30w_50_5e-3_decay2000.txt',para1_record)

plt.figure()
plt.plot(range(0, len(para2_record) * 1000, 1000), para2_record, label="para2", color='b')
plt.axhline(y=0.01, color='r', linestyle='--', label='True $para =0.01$')  # 真值
plt.title("Training of para")
plt.xlabel("Iterations")
plt.ylabel("parameters value")

plt.savefig('para0.01_30w_50_1e-3_decay1000.png')

np.savetxt('para0.01_record_30w_50_1e-3_decay1000.txt',para2_record)
