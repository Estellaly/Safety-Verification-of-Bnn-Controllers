import numpy as np
import tensorflow as tf
from edward.models import Normal
import edward as ed
ed.set_seed(980297)
latent = 16
class BNN():
   
    layers=[latent]
    units=latent
    def __init__(self, n_dim ,u_dim,n_iter,sc,n_sample):
        self.n_dim=n_dim
        self.u_dim=u_dim
        self.n_iter = n_iter
        self.sc = sc
        self.n_sample = n_sample
        
        self.__build__()
        
    def __build__(self):
        sc=self.sc
        self.x = tf.placeholder(tf.float32, [None, self.n_dim])
        self.y_ph = tf.placeholder(tf.float32, [None, self.u_dim])
        self.feed=dict()
        self.W_fc1 = Normal(loc=tf.zeros([self.n_dim, self.units]), scale=tf.ones([self.n_dim, self.units])*sc)
        self.b_fc1 = Normal(loc=tf.zeros(self.units), scale=tf.ones(self.units)*sc)
        self.h_fc1 = tf.nn.relu(tf.matmul(self.x, self.W_fc1) + self.b_fc1)
        self.w = Normal(loc=tf.zeros([self.units, self.u_dim]), scale=tf.ones([self.units, self.u_dim])*sc)
        self.b = Normal(loc=tf.zeros(self.u_dim), scale=tf.ones(self.u_dim)*sc)

        self.y = Normal(loc=tf.matmul(self.h_fc1, self.w) + self.b,scale=tf.ones([self.u_dim])*sc)

        self.qW_fc1 = Normal(loc=tf.Variable(tf.random_normal([self.n_dim, self.units]),name='w1m'),
                        scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.n_dim, self.units]),name='w1d')))
        self.qb_fc1 = Normal(loc=tf.Variable(tf.random_normal([self.units]),name='b1m'),
                        scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.units])),name='b1d'))

        self.qw = Normal(loc=tf.Variable(tf.random_normal([self.units, self.u_dim]),name='w2m'),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.units, self.u_dim])),name='w2d'))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([self.u_dim]),name='b2m'),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.u_dim])),name='b2d'))

        #self.out=out
        self.feed={self.W_fc1: self.qW_fc1, self.b_fc1: self.qb_fc1, self.w: self.qw, self.b: self.qb}
        self.inference = ed.KLqp(self.feed, data={self.y: self.y_ph})
        # 设置优化器
        optimizer = tf.train.AdamOptimizer(0.001)
        self.inference.initialize(n_iter=self.n_iter, n_print=200,n_samples=self.n_sample,optimizer=optimizer)

        # self.inference.initialize(n_iter=self.n_iter, n_print=200,n_samples=self.n_sample)

        gpu_options = tf.GPUOptions(allow_growth=True)
        tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        tf.global_variables_initializer().run()
    def train(self,X,y,pathW):
        losses =[]
        for i in range(self.inference.n_iter):
            info_dict = self.inference.update(feed_dict={self.x: X, self.y_ph:y})
            losses.append(info_dict['loss'])
            if i%10000 ==0:
                self.saveWeights(pathW+str(i)+"_"+str(self.sc)+'_'+str(self.n_sample)+'.npz')
            self.inference.print_progress(info_dict)
        return losses
    def test(self,X,y):
        n_samples = 5
        mes_list = []

        for _ in range(n_samples):
            wfc1_samp = self.qW_fc1.sample()
            bfc1_samp = self.qb_fc1.sample()
            w_samp = self.qw.sample()
            b_samp = self.qb.sample()
            h_fc1 = tf.nn.relu(tf.matmul(X, wfc1_samp) + bfc1_samp)
            y_=tf.matmul(h_fc1, w_samp) + b_samp
            y_=y_.eval()
            mes=np.mean((y_-y)**2)
            mes_list.append(mes)
            print(y_[0], y[0])
        print('mes:',np.mean(mes_list))

        return np.mean(mes_list)

    def predict(self,x):
        x=np.float32(x)
        sample_num=10
        y=[]
        wfc1_samp = self.qW_fc1.sample()
        bfc1_samp = self.qb_fc1.sample()
        w_samp = self.qw.sample()
        b_samp = self.qb.sample()
        h_fc1 = tf.nn.relu(tf.matmul(x, wfc1_samp) + bfc1_samp)
        y_ = tf.matmul(h_fc1, w_samp) + b_samp
        return y_.eval()
    def criticism(self,X_test,y_test):
        y_post = ed.copy(self.y, {self.W_fc1: self.qW_fc1, self.b_fc1: self.qb_fc1, self.w: self.qw, self.b: self.qb})
        print("Mean squared error on test data:")
        print(ed.evaluate('mean_squared_error', data={self.x: X_test, y_post: y_test}))

        print("Mean absolute error on test data:")
        print(ed.evaluate('mean_absolute_error', data={self.x: X_test, y_post: y_test}))

    def samples(self,sampelTimes):
        wfc1_samp=[]
        bfc1_samp=[]
        w_samp=[]
        b_samp=[]
        for i in range(sampelTimes):
            wfc1_samp.append(self.qW_fc1.sample())
            bfc1_samp.append(self.qb_fc1.sample())
            w_samp.append(self.qw.sample())
            b_samp.append(self.qb.sample())
        wfc1_samp = (tf.reshape(wfc1_samp,[sampelTimes,self.n_dim,-1])).eval()
        bfc1_samp = (tf.reshape(bfc1_samp,[sampelTimes,-1])).eval()
        w_samp = (tf.reshape(w_samp,[sampelTimes,-1,self.u_dim])).eval()
        b_samp = (tf.reshape(b_samp,[sampelTimes,-1])).eval()
        return wfc1_samp,bfc1_samp,w_samp,b_samp
    def MeanAndStd(self):
        wfc1_m=(self.qW_fc1.loc).eval()
        bfc1_m=(self.qb_fc1.loc).eval()
        w_m=(self.qw.loc).eval()
        b_m=(self.qb.loc).eval()
        wfc1_d=(self.qW_fc1.variance()).eval()
        bfc1_d=(self.qb_fc1.variance()).eval()
        w_d=(self.qw.variance()).eval()
        b_d=(self.qb.variance()).eval()
        return wfc1_m,bfc1_m,w_m,b_m,wfc1_d,bfc1_d,w_d,b_d
    def saveWeights(self,nameIndex):
        path = nameIndex
        wfc1_m,bfc1_m,w_m,b_m,wfc1_d,bfc1_d,w_d,b_d = self.MeanAndStd()
        np.savez_compressed(path,mW_0=wfc1_m,mb_0=bfc1_m,mW_1=w_m,mb_1=b_m,dW_0=wfc1_d,db_0=bfc1_d,dW_1=w_d,db_1=b_d)