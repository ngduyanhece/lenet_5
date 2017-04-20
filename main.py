import numpy as np
from flask import Flask, jsonify, render_template, request
from six.moves import cPickle
from model import CNN
from model import HidenLayer
from model import ConvPoolLayer
# from mnist import model
#
#
# x = tf.placeholder("float", [None, 784])
# sess = tf.Session()
# # restore trained data
# with tf.variable_scope("regression"):
#     y1, variables = model.regression(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/regression.ckpt")
#
#
# with tf.variable_scope("convolutional"):
#     keep_prob = tf.placeholder("float")
#     y2, variables = model.convolutional(x, keep_prob)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/convolutional.ckpt")
#
f = open("mnist/data/mnist_cnn.ckpt", 'rb')
mnist_model = cPickle.load(f)
f.close()
def regression(input):
    return mnist_model.predict(input)


# def convolutional(input):
#     return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.float32))).reshape(1,1,28,28)
    output1 = regression(input)
    output1 = output1.eval()
    print output1
    number = output1[0]
    return jsonify(results=number)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
