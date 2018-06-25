# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:26:55 2018

@author: Ali Nasr
"""


# import important
import tensorflow as tf
import cv2
import numpy as np



width_img = 224
hight_img = 224

width_img_cam = 352
hight_img_cam = 288

frame = np.zeros((hight_img, width_img, 1), np.uint8)
windowName = 'cam'
cv2.namedWindow(windowName)
l = 0

hm_epochs = 20
number_of_data = 20
n_classes = 2
batch_size = 1

x = tf.placeholder('float', [None, width_img * hight_img])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def mouse(event, x, y, flags, param):
    global l

    if event == cv2.EVENT_RBUTTONDOWN:
        l = 1  # it actually zero it will be changed in capture_ dataset function
    if event == cv2.EVENT_LBUTTONDOWN:
        l = 2  # it actually 1 it will be changed in capture_ dataset function


# bind the callback function to window
cv2.setMouseCallback(windowName, mouse)


def capture_dataset():
    global l
    img = np.zeros((number_of_data, hight_img, width_img))
    label = np.zeros(number_of_data).astype(int)
    i = 0

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    cap.set(3, width_img_cam);
    cap.set(4, hight_img_cam);

    print(cap.get(3))
    print(cap.get(4))

    while (True):
        ret, camera_img = cap.read()
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(camera_img, (width_img, hight_img))
        cv2.imshow(windowName, frame)

        if l == 1:
            label[i] = 0;
            img[i] = frame
            l = 0
            i += 1
        elif l == 2:
            label[i] = 1
            img[i] = frame
            l = 0
            i += 1

        if i == number_of_data:
            break

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()

    for j in range(number_of_data):
        cv2.imwrite("save/" + str(j) + "_" + str(label[j]) + "_" + "images.jpg", img[j],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    label = np.eye(n_classes)[label.reshape(-1)]
    return img, label


def conv_2D_(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def Maxpooling_2D(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'Weights_Conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'Weights_Conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'Weights_FC': tf.Variable(tf.random_normal([56 * 56 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'Biase_Conv1': tf.Variable(tf.random_normal([32])),
              'Biase_Conv2': tf.Variable(tf.random_normal([64])),
              'Biase_FC': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, hight_img, width_img, 1])

    conv1 = conv_2D_(x, weights['Weights_Conv1'])
    conv1 = Maxpooling_2D(conv1)

    conv2 = conv_2D_(conv1, weights['Weights_Conv2'])
    conv2 = Maxpooling_2D(conv2)

    fc = tf.reshape(conv2, [-1, 56 * 56 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['Weights_FC']) + biases['Biase_FC'])

    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    images1, labels1 = capture_dataset()

    prediction = convolutional_neural_network(x)
 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))


    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for k in range(number_of_data):
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = images1[k].reshape(1, width_img * hight_img)
                epoch_y = labels1[k]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # please customize the directory for your project    
        saver.save(sess, '/home/ali/PycharmProjects/test1/saved/my_test_model')

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accyracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


def feed_for():
    global l
    img = np.zeros((number_of_data, hight_img, width_img))
    label = np.zeros(number_of_data).astype(int)
    i = 0

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    cap.set(3, width_img_cam);
    cap.set(4, hight_img_cam);

    print(cap.get(3))
    print(cap.get(4))

    sess = tf.Session()
    # please customize the directory for your project
    saver = tf.train.import_meta_graph('/home/ali/PycharmProjects/test1/saved/my_test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/ali/PycharmProjects/test1/saved/./'))

    graph = tf.get_default_graph()
    #w1 = graph.get_tensor_by_name("w1:0")
    #w2 = graph.get_tensor_by_name("w2:0")

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("add_1:0")
    forward = tf.nn.softmax(logits=op_to_restore)


    while (True):
        ret, camera_img = cap.read()
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(camera_img, (width_img, hight_img))
        cv2.imshow(windowName, frame)

        epoch_x = frame.reshape(1, width_img * hight_img)
        feed_dict = {x: epoch_x}

        sess.run(tf.global_variables_initializer())
        #print(forward.eval(feed_dict))
        print(sess.run(forward, feed_dict))


        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()


def main():

    train_neural_network(x)
    feed_for()


main()
