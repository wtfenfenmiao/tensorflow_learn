#coding:utf-8
import tensorflow as tf
import numpy as np
import input_data

mnist=input_data.read_data_sets("E:\\tensorflow_learn\MNIST_helloWorld\MNIST_data",one_hot=True)

def Weight_Variable(shape):
    initial=tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)

def bias_Variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def accuracy(test_x,test_y):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:test_x,keep_prob:1})
    true_pre=tf.equal(tf.argmax(test_y,1),tf.argmax(y_pre,1))
    pre_accuracy=tf.reduce_mean(tf.cast(true_pre,tf.float32))
    result=sess.run(pre_accuracy)   #开始忘了写这句话，血崩
    return result

if __name__=="__main__":
    xs=tf.placeholder(tf.float32,[None,784])
    ys=tf.placeholder(tf.float32,[None,10])
    keep_prob=tf.placeholder(tf.float32)

    x_image=tf.reshape(xs,[-1,28,28,1])

    W_conv1=Weight_Variable([5,5,1,32])
    b_conv1=bias_Variable([32])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h1_pool=max_pooling(h_conv1)

    W_conv2=Weight_Variable([5,5,32,64])
    b_conv2=bias_Variable([64])
    h_conv2=tf.nn.relu(conv2d(h1_pool,W_conv2)+b_conv2)
    h2_pool=max_pooling(h_conv2)

    x_fc1=tf.reshape(h2_pool,[-1,7*7*64])
    w_fc1=Weight_Variable([7*7*64,1024])
    b_fc1=bias_Variable([1024])
    h_fc1=tf.nn.relu(tf.matmul(x_fc1,w_fc1)+b_fc1)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    w_fc2=Weight_Variable([1024,10])
    b_fc2=bias_Variable([10])
    prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    saver=tf.train.Saver()
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        xs_batch,ys_batch=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:xs_batch,ys:ys_batch,keep_prob:0.5})
        if i%50 == 0:
            print (accuracy(mnist.test.images,mnist.test.labels))
    savepath=saver.save(sess,'mynet\cnn.ckpt')
    print ("save to path:",savepath)       #这样就把这个网络的各个参数存下来了。但是注意，存下来的是参数，没有网络结构！还得再定义网络结构，还得同名......
    #https://blog.csdn.net/tan_handsome/article/details/79303269  随便搜，tensorflow保存模型得到的，网络和参数都能存。最后我们肯定用这个，怎么方便怎么来，嗯


