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



def net_u(xu, weights, biases):

    u = neural_net(xu, weights, biases)
    return u


def net_f(xf, weights_u, biases_u, para1, para2,para3):

    u = net_u(xf, weights_u, biases_u)
    u_x = tf.gradients(u, xf)[0]
    u_xx = tf.gradients(u_x, xf)[0]
    f = (para1 * xf** 2)/ (xf ** 2+para2) -xf + para3

    return f * u_x + 1/8 * u_xx


def net_drift(xf, para1, para2,para3):
    f=((para1 * xf ** 2) / (xf ** 2 + para2) -xf + para3)
    return f


def u_exact(x):
    return (x1 - x) * (-x0 + x)


##left side
def f_exact(x):
    return -1


###true drift
def g_exact(x):
    return ((6*x**2)/(x**2+10) -x +0.4)

x0, x1 = 0,1.48971

layers = [1] + 2 * [32] + [1]
L = len(layers)

weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]

para1=tf.Variable( 0.1,dtype=tf.float64)
para2=tf.Variable( 0.1,dtype=tf.float64)
para3=tf.Variable( 0.1,dtype=tf.float64)


xf = np.reshape(np.linspace(x0,x1,101),[-1,1])
xf_tf = tf.to_double(xf)

f_target = f_exact(xf)

f_pred= net_f(xf_tf, weights_u, biases_u, para1,para2,para3)


u_opt=np.loadtxt('./101-ut0.txt')
N_ob=5
choose_ob=np.linspace(0,100,N_ob,dtype=int)
u_ob=u_opt[choose_ob][:,None]
xf = np.reshape(np.linspace(x0,x1,N_ob),[-1,1])
u_ob_NN=net_u(xf, weights_u, biases_u)

loss_ob=tf.reduce_mean(tf.square(u_ob-u_ob_NN))

loss_f=tf.reduce_mean(tf.square(f_target - f_pred))


loss =loss_f+100*loss_ob
optimizer_Adam = tf.train.AdamOptimizer(5e-3)
train_op_Adam = optimizer_Adam.minimize(loss)


x_test=np.reshape(np.linspace(x0,x1,101),[-1,1])
u_test = u_exact(x_test)
drift_test=g_exact(x_test)
min_loss = 1e16
saver = tf.train.Saver(max_to_keep=1000)
savedir='./met_inverse_driftpara_weight5'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    u_pred=net_u(x_test,weights_u,biases_u)
    drift_pred=net_drift(x_test,para1,para2,para3)
    f_p = net_f( tf.to_double(x_test), weights_u, biases_u, para1,para2,para3)
    loss_record = []
    loss_f_record = []
    loss_ob_record = []
    for i in range(45000+1):
        sess.run(train_op_Adam)
        if i % 1000 == 0:
            temp_loss=sess.run(loss)
            (loss_result, loss_pde_result, loss_ob_result) = sess.run([loss,
                    loss_f, loss_ob])
            (para1_train,para2_train,para3_train)=sess.run([para1,para2,para3])
            if temp_loss<min_loss:
                min_loss=temp_loss
                ut_opt=np.reshape(sess.run(u_pred),[-1,1])
                ft_opt=np.reshape(sess.run(f_p),[-1,1])
                drift_opt=np.reshape(sess.run(drift_pred),[-1,1])
                final_para1 = para1_train
                final_para2 = para2_train
                final_para3 = para3_train

                u_ob_NN_opt=np.reshape(sess.run(u_ob_NN),[-1,1])

                i_opt=i

                save_path = saver.save(sess, savedir + '/metinver_driftpara_best_model5.ckpt')
                (weights_u_np, biases_u_np) = sess.run([weights_u, biases_u])
                sample_list = {
                    "weights_u": weights_u_np,
                    "biases_u": biases_u_np,
                    "para1": final_para1,
                    "para2": final_para2,
                    "para3": final_para3
                }

                file_name = './met_inverse_driftpara_weight5/metinver_driftpara_hyper_best_model5.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()
            loss_record.append(temp_loss)
            ut0=np.reshape(sess.run(u_pred),[-1,1])
            drift_result=np.reshape(sess.run(drift_pred),[-1,1])

            print ('  %d   total loss: %8.2e pde loss: %8.2e u loss: %8.2e drift error: %8.2e para1: %8.2e para2: %8.2e  para3: %8.2e  para4: %8.2e' % (i,  temp_loss,loss_pde_result,loss_ob_result,error_drift,para1_train,para2_train,para3_train,para4_train) )




    print("para1 =", final_para1)
    print("para2 =", final_para2)
    print("para3 =", final_para3)


print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)
#np.savetxt("./result/f-mat.csv", f_target - f_pred, delimiter=",")
#np.savetxt("./result/u.csv", ut0, delimiter=",")

# plt.plot(x_test,drift_opt,'r')
# plt.plot(x_test,g_exact(x_test),'k--')
# plt.xlabel('x')
# plt.ylabel('MET_inverse_driftpara')
# plt.legend(['predicted solution','true solution'] ,frameon=False)
# plt.title('Genetic regulatory system MET_inverse_driftpara')
# plt.show()