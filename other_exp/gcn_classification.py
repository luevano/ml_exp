import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1.layers import dense


# Disable eager execution (evaluate tf operations instantly instead of having
# to build a graph) so placeholder() can work.
tf.compat.v1.disable_eager_execution()


n_nodes = 50
n_features = 50
n_labels = 10

X = placeholder(tf.float64, shape=(None, n_nodes, n_features))
A = placeholder(tf.float64, shape=(None, n_nodes, n_nodes))
Y_truth = placeholder(tf.float64, shape=(None, n_labels))

# Function for implementation of H⁽l+1)=sigma(A(AH^lW^l)+ b^l).
# With the bias term given by the tf dense layer.
def graph_conv(_X, _A, O):
    """
    Equation of graph convolution.
    _X: vector X. Nodes.
    _A: adjacency matrix. Edges or path.
    """
    out = dense(_X, units=O, use_bias=True)
    out = tf.matmul(_A, out)
    out = tf.nn.relu(out)

    return out

X_new = graph_conv(X, A, 32)
print(X_new)

gconv1 = graph_conv(X, A, 32)
gconv2 = graph_conv(gconv1, A, 32)
gconv3 = graph_conv(gconv2, A, 32)

Y_pred = tf.nn.softmax(dense(gconv3, units=n_labels, use_bias=True), axis=2)
print(Y_pred)

Y_pred = tf.reshape(Y_pred, [-1])
loss = tf.reduce_mean(Y_truth*tf.math.log(Y_pred + 1.0 ** -5))

print(loss)


print(tf.__version__)
