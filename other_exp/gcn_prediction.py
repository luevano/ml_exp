import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.layers import dense

tf.compat.v1.disable_eager_execution()

n_nodes = 50
n_features = 50
X = placeholder(tf.float64, [None, n_nodes, n_features])
A = placeholder(tf.float64, [None, n_nodes, n_nodes])
Y_truth = placeholder(tf.float64, [None, ])


def graph_conv(_X, _A, O):
    out = dense(_X, units=O, use_bias=True)
    out = tf.matmul(_A, out)
    out = tf.nn.relu(out)

    return out


def readout_nw(_X, O):
    """
    Node-wise summation implementation.
    _X: final node embeddings.
    O: desired output dimension.
    """
    out = dense(_X, O, use_bias=True)
    out = tf.reduce_sum(out, axis=1)
    out = tf.nn.relu(out)

    return out


def readout_gg(_X, X, O):
    """
    Graph Gathering implementation. (The none shown in the equation)
    _X: final node embeddings.
    X: initial node features.
    O: desired output dimension.
    """
    val1 = dense(tf.concat([_X, X], axis=2), O, use_bias=True)
    val1 = tf.nn.sigmoid(val1)
    val2 = dense(_X, O, use_bias=True)

    out = tf.multiply(val1, val2)
    out = tf.reduce_sum(out, axis=1)
    out = tf.nn.relu(out)

    return out


gconv1 = graph_conv(X, A, 32)
gconv2 = graph_conv(gconv1, A, 32)
gconv3 = graph_conv(gconv2, A, 32)

graph_feature = readout_gg(gconv3, gconv1, 128)
print("\t\tNode-wise summation implementation result.")
print(graph_feature)

Y_pred = dense(graph_feature, 128, use_bias=True, activation=tf.nn.relu)
Y_pred = dense(Y_pred, 128, use_bias=True, activation=tf.nn.tanh)
Y_pred = dense(Y_pred, 1, use_bias=True, activation=None)
print(Y_pred)

Y_pred = tf.reshape(Y_pred, shape=[-1])
Y_truth = tf.reshape(Y_truth, shape=[-1])
loss = tf.reduce_mean(tf.pow(Y_truth - Y_pred, 2))
print(loss)
