#coding:utf-8
import tensorflow as tf
import numpy as np
import input_data

mnist=input_data.read_data_sets('E:\\tensorflow_learn\MNIST_helloWorld\MNIST_data',one_hot=True)

def add_layer(input_data,in_size,out_size,activate_function):
    W=tf.Variable(tf.random_normal([in_size,out_size]))
    b=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(input_data,W)+b
    if activate_function is None:
        output = Wx_plus_b
    else:
        output = activate_function(Wx_plus_b)
    return output

def compute_accuracy(test_x,test_y):
    global prediction  #用global的作用：告诉函数，这个东西虽然这里没看见，但是外面是有的，这是个全局变量，虽然你现在没看见，但是用的时候就看见了
    #如果你想要为一个定义在函数外的变量赋值，那么你就得告诉Python这个变量名不是局部的，而是全局的。我们使用global语句完成这一功能。没有global语句，是不可能为定义在函数外的变量赋值的。
    #你可以使用定义在函数外的变量的值（假设在函数内没有同名的变量）。然而，我并不鼓励你这样做，并且你应该尽量避免这样做，因为这使得程序的读者会不清楚这个变量是在哪里定义的。使用global语句可以清楚地表明变量是在外面的块定义的。
    y_pre = sess.run(prediction,feed_dict = {xs:test_x})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(test_y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #result = sess.run(accuracy,feed_dict={xs:test_x,ys:test_y})
    result = sess.run(accuracy)    #这个感觉，不需要，feed_dict......事实证明也确实不需要
    return result

if __name__=="__main__":
    xs=tf.placeholder(tf.float32,[None,784])
    ys=tf.placeholder(tf.float32,[None,10])

    prediction=add_layer(xs,784,10,activate_function=tf.nn.softmax)

    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i%50 == 0:
            print (compute_accuracy(mnist.test.images,mnist.test.labels))


