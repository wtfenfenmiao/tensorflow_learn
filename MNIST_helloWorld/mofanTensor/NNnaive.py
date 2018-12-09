#coding:utf-8
#被莫烦圈粉。这个写注释很舒服，嗯
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input_data,in_size,out_size,active_function):
    W=tf.Variable(tf.random_normal([in_size,out_size]))
    b=tf.Variable(tf.zeros([1,out_size])+0.1)
    wx_plus_b=tf.matmul(input_data,W)+b   #matmul：矩阵相乘    multiply:元素相乘
    if active_function is None:
        output=wx_plus_b
    else:
        output=active_function(wx_plus_b)
    return output



if __name__=="__main__":
    x=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]   #这个newaxis相当于None。这个[:,np.newaxis]作用相当于reshape。行向量弄成了一个列向量
    noise=np.random.normal(0,0.05,x.shape).astype(np.float32)
    y=np.square(x)-0.5+noise

    fig=plt.figure()     #我的感觉，就是弄了个画布，嗯
    ax=fig.add_subplot(2,2,3)    #返回一个对象。可以在一个大图上做若干子图。
    #这三个数的意思。前两个数是将fig分成多少块，比如2，2就是4块，2*2。然后第三个数是选取第几块。比如2*2的时候，如果是1，就是左上角那块，2就是右上角，3就是左下角......
    # 111就是只有一块了
    ax.scatter(x,y)   #散点图。
    plt.ion()
    plt.show()    #这个plt.show，如果不加上一句，就会暂停，画完一张停在这里(阻塞模式)，你操作了才会画下一张。上一个ion开启了交互模式

    xs=tf.placeholder(tf.float32,[None,1])   #有多少条数据不重要。重要的是这个数据长什么样
    ys=tf.placeholder(tf.float32,[None,1])

    l1=add_layer(xs,1,10,active_function=tf.nn.relu)
    prediction=add_layer(l1,10,1,active_function=None)

    print (ys-prediction)
    print (tf.square(ys-prediction))
    print (tf.reduce_sum(tf.square(ys-prediction)))
    print(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))            #损失函数是均方误差
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init=tf.global_variables_initializer()

    sess=tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x,ys:y})
        if i%50 == 0:
            #print(sess.run(loss,feed_dict={xs:x,ys:y}))
            pre=sess.run(prediction,feed_dict={xs:x})
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines=ax.plot(x,pre,'r-',lw=5)   #线是红色，线粗是5
            plt.pause(0.2)





