#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.special as sci
np.random.seed(1234)
tf.set_random_seed(1234)
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pickle



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
    u = neural_net(xu, weights, biases)
    return u


def net_f(xf, weights_u,biases_u,weights_f,biases_f):

    u = net_u(xf, weights_u, biases_u)
    u_x = tf.gradients(u, xf)[0]
    u_xx = tf.gradients(u_x, xf)[0]
    g= neural_net(xf, weights_f, biases_f)

    return g * u_x+1/8*u_xx

def u_exact(x):
    return (x1-x) * (-x0+x)

##left side
def f_exact(x):
    return 0
###true drift
def g_exact(xf):
    return ((6*xf**2)/(xf**2+10) -xf +0.4)

x0, x1 = 0,1.48971



layers = [1] + 4* [32] + [1]
L = len(layers)

weights_u = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases_u = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]


layers_f = [1] + 4 * [32] + [1]
L_f = len(layers_f)
weights_f = [xavier_init([layers_f[l], layers_f[l+1]]) for l in range(0, L_f-1)]
biases_f = [tf.Variable( tf.zeros((1, layers_f[l+1]),dtype=tf.float64)) for l in range(0, L_f-1)]

xf = np.reshape(np.linspace(x0,x1,101),[-1,1])

xf_tf = tf.to_double(xf)

f_target = f_exact(xf)

f_pred= net_f(xf_tf, weights_u, biases_u, weights_f, biases_f)

u_opt=np.loadtxt('./1001-ut0-ep.txt')
N_ob=21
choose_ob=np.linspace(0,1000,N_ob,dtype=int)
u_ob=u_opt[choose_ob][:,None]
xf = np.reshape(np.linspace(x0,x1,N_ob),[-1,1])
u_ob_NN=net_u(xf, weights_u, biases_u)
loss_ob=tf.reduce_mean(tf.square(u_ob-u_ob_NN))

loss_f=tf.reduce_mean(tf.square(f_target - f_pred))


loss =loss_f+100*loss_ob
optimizer_Adam = tf.train.AdamOptimizer(5e-4)
train_op_Adam = optimizer_Adam.minimize(loss)


x_test=np.reshape(np.linspace(x0,x1,101),[-1,1])
u_test = u_exact(x_test)
drift_test=g_exact(x_test)
min_loss = 1e16

saver = tf.train.Saver(max_to_keep=1000)
savedir='./ep_inverse_drift_weight21'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    u_pred=net_u(x_test,weights_u,biases_u)
    drift_pred=net_u(x_test,weights_f,biases_f)
    f_p= net_f( tf.to_double(x_test), weights_u, biases_u, weights_f, biases_f)
    loss_record = []
    loss_f_record = []
    loss_ob_record = []
    for i in range(50000+1):
        sess.run(train_op_Adam)
        if i % 1000 == 0:
            temp_loss=sess.run(loss)
            (loss_result, loss_pde_result, loss_ob_result) = sess.run([loss,
                    loss_f, loss_ob])
            if temp_loss<min_loss:
                min_loss=temp_loss
                ut_opt=np.reshape(sess.run(u_pred),[-1,1])
                ft_opt=np.reshape(sess.run(f_p),[-1,1])
                drift_opt=np.reshape(sess.run(drift_pred),[-1,1])

                u_ob_NN_opt=np.reshape(sess.run(u_ob_NN),[-1,1])

                i_opt=i
                save_path = saver.save(sess, savedir + '/epinver_drift_best_model21.ckpt')
                (weights_u_np, biases_u_np, weights_f_np, biases_f_np) = sess.run(
                    [weights_u, biases_u, weights_f, biases_f])
                sample_list = {"weights_u": weights_u_np, "biases_u": biases_u_np, "weights_f": weights_f_np,
                               "biases_f": biases_f_np}

                file_name = './ep_inverse_drift_weight21/epinver_drift_hyper_best_model21.pkl'
                with open(file_name, "wb") as open_file:
                    pickle.dump(sample_list, open_file)
                open_file.close()
            loss_record.append(temp_loss)
            ut0=np.reshape(sess.run(u_pred),[-1,1])
            drift_result=np.reshape(sess.run(drift_pred),[-1,1])
            error_u=np.linalg.norm(ut0-u_test,2)/np.linalg.norm(u_test,2)
            error_drift=np.linalg.norm(drift_result-drift_test,2)/np.linalg.norm(drift_test,2)
            print ('  %d   total loss: %8.2e pde loss: %8.2e u loss: %8.2e drift error: %8.2e ' % (i,  temp_loss,loss_pde_result,loss_ob_result,error_drift) )
#             print ('  %d    %8.2e  %8.2e' % (i,  temp_loss, error_u) )

print  ('Loss is %10.5e' % min_loss)
print ('Best iteration is %d' % i_opt)
#np.savetxt("./result/f-mat.csv", f_target - f_pred, delimiter=",")
#np.savetxt("./result/u.csv", ut0, delimiter=",")


# plt.plot(x_test,drift_opt,'r--')
# plt.plot(x_test,g_exact(x_test),'k')
# plt.xlabel('x')
# plt.ylabel('EP-inverse_drift')
# plt.legend(['predicted solution','true solution'] ,frameon=False)
# plt.title('Genetic regulatory system EP_inverse_drift')
# plt.show()