import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1,M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape,poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)
def error_rate(Y,T):
    return np.mean(Y!=T)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    y_ind = np.zeros((N,K))
    for i in xrange(N):
        y_ind[i,int(y[i])] = 1
    return y_ind

class HidenLayer(object):
    def __init__(self,M1,M2,an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W,b = init_weight_and_bias(M1,M2)
        self.W = theano.shared(W,'W_%s' % self.id)
        self.b = theano.shared(b,'b_%s' % self.id)
        self.params = [self.W,self.b]
    def forward(self,X):
        return T.nnet.relu(X.dot(self.W) + self.b)

class ConvPoolLayer(object):
    def __init__(self,mi,mo,fw=5,fh=5,poolsz=(2,2)):
        sz = (mo,mi,fw,fh)
        W0 = init_filter(sz,poolsz)
        self.W = theano.shared(W0)
        b0 = np.zeros(mo,dtype=np.float32)
        self.b = theano.shared(b0)
        self.poolsz = poolsz
        self.params = [self.W,self.b]
    def forward(self,X):
        conv_out = conv2d(input=X,filters=self.W)
        pool_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsz,
            ignore_border=True
        )
        return T.nnet.relu(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
class CNN(object):
    def __init__(self,convpool_layer_sizes,hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
    def train(self, X, Y, lr=10e-8, mu=0.99, reg=10e-8, decay=0.99999, eps=10e-3, batch_sz=100, epochs=20, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        eps = np.float32(eps)

        #make validation set
        X,Y = shuffle(X,Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X,Y = X[:-1000], Y[:-1000]
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        Y_ind = y2indicator(Y)
        Y_valid_ind = y2indicator(Y_valid)
        #initialize convpool layers
        N,c, width, height = X.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        for mo,fw,fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi,mo,fw,fh)
            self.convpool_layers.append(layer)
            outw = (outw - fw + 1) / 2
            outh = (outh - fh + 1) / 2
            mi = mo
        #initialize the fully connected layers
        K = len(set(Y))
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HidenLayer(M1,M2,count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        # logistic regression layer
        W, b = init_weight_and_bias(M1,K)
        self.W = theano.shared(W,'W_logreg')
        self.b = theano.shared(b,'b_logreg')

        # collect params for latter use
        self.params = [self.W,self.b]
        for c in self.convpool_layers:
            self.params += c.params
        for h in self.hidden_layers:
            self.params + h.params
        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in self.params]
        cache = [theano.shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in self.params]
        # setup theano functions and variables
        th_X = T.tensor4('X',dtype='float32')
        th_Y = T.matrix('Y')
        P_Y = self.forward(th_X)
        rcost = T.sum([(p*p).sum() for p in self.params])
        cost = -(th_Y*T.log(P_Y)).sum() + rcost
        prediction = self.predict(th_X)
        cost_prediction_op = theano.function(
            inputs = [th_X,th_Y],
            outputs = [cost,prediction]
        )
        # updates
        updates = [
            (p, p + mu*dp - lr*T.grad(cost,p)) for p, dp in zip(self.params, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost,p)) for p, dp in zip(self.params, dparams)
        ]
        train_op = theano.function(
            inputs = [th_X,th_Y],
            updates = updates
        )
        n_batches = N / batch_sz
        costs = []
        for i in xrange(epochs):
            for j in xrange(n_batches):
                X_batch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Y_batch = Y_ind[j*batch_sz:(j*batch_sz + batch_sz)]
                train_op(X_batch,Y_batch)
                if j % 20 == 0:
                    c,p = cost_prediction_op(X_valid,Y_valid_ind)
                    costs.append(c)
                    e = error_rate(Y_valid,p)
                    print "i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e
        if show_fig:
            plt.plot(costs)
            plt.show()
    def forward(self,X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z = Z.flatten(ndim=2)
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)
    def predict(self,X):
        P_Y = self.forward(X)
        return T.argmax(P_Y,axis=1)
