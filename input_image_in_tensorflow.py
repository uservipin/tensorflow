# dense layer means fully conneted layer

# 2**3 means 2 raises to power to 3




print(3*2**3)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow   as tf
from sklearn.metrics import confusion_matrix
# from PIL import Image
# from skimage.util.montage import montage2d
import tflearn
from tensorflow.examples.tutorials.mnist import input_data



data=input_data.read_data_sets("data/MNIST/",one_hot=True)


print("shape of image in traning data set{}".format(data.train.images.shape))
print("shape of classes in traning data set{}".format(data.train.labels.shape))
print("shape of image in test data set{}".format(data.test.images.shape))
print("shape of classes in test data set{}".format(data.test.labels.shape))
print("shape of image in velidation data set{}".format(data.validation.images.shape))
print("shape of lables in velidation data set{}".format(data.validation.labels.shape))



# for i in range (1,5):
#     sample=data.train.images[i].reshape(28,28)
#     print("show")
#     plt.imshow(sample)
#     print("sample")
#     plt.show(i)








# sample = data.train.images[2].reshape(28, 28)
# plt.imshow(sample)
# plt.title('sample_image')
# plt.show()


# function to show mmontage
imags=data.train.images[0:500]
print("image size",imags.shape)
print(imags)
# montage_images=np.zeros([100,28,28])
# for i in range(len(imags)):
#     montage_images[i]=imags[i].reshape(28,28)
# plt.imshow(montage2d(montage_images),cmap='gray')
# plt.title('sample of output image')
# plt.axis('off')
# plt.show()

# input images
x= tf.placeholder(tf.float32,shape=[None,784])


# input class
y_ =tf.placeholder(tf.float32,shape=[None,10])

# model
x_input=tf.reshape(x,[-1,28,28,1],name='input')


conv_layer=tflearn.layers.conv.conv_2d(x_input,nb_filter=18,filter_size=5,strides=[1,3,3,1],
                                       padding='same',activation='relu',regularizer='L2',name='conv_layer_1')

out_layer_1= tflearn.layers.conv.max_pool_2d(conv_layer,2)

conv_layer_2= tflearn.layers.conv.conv_2d(out_layer_1,nb_filter=18,filter_size=5,strides=[1,3,3,1],
                                          padding='same',activation='relu',regularizer='L2',
                                          name='conv_layer_2')

out_layer_2=tflearn.layers.conv.max_pool_2d(conv_layer_2,2)

fc1=tflearn.layers.core.fully_connected(out_layer_2,1024,activation='relu')
fc1_dropout=tflearn.layers.core.dropout(fc1,0.8)
y_predicted=tflearn.layers.core.fully_connected(fc1_dropout,10,activation='softmax',name='output')




print("shape of input :{}".format(x_input.get_shape().as_list()))
print ("shape of first convolution layer :{}".format(out_layer_1.get_shape().as_list))
print ("shape of second convolution layer :{}".format(out_layer_2.get_shape().as_list))
print ("shape of  fully connecting  layer :{}".format(fc1.get_shape().as_list))
print ("shape of output layer :{}".format(y_predicted.get_shape().as_list))


# loss function
cross_entropy=tf.reduce_mean(tf.reduce_sum(y_ *tf.log(y_predicted),reduction_indices=[1]))
print("cross")
# optimizer
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
print("train")
# calculating accuracy of our model
correct_prediction=tf.equal(tf.argmax(y_predicted,1),tf.arg_max(y_,1))
print("correct ")
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("accuracy")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

# sess=tf.Session()
init =tf.global_variables_initializer()
# init= tf.initialize_local_variables()
sess.run(init)

# grab the default graph
g=tf.get_default_graph()

# every operation in our graph
# [op.name for op in g.get_operations()]

# no. of itration
epoch =100000
batch_size=5
print('batch size')

x_batch,y_batch=data.train.next_batch(batch_size)

print(x_batch.shape,y_batch.shape)
print("\n")
print(y_batch)


for i in range(epoch):

    # batch wise traning
    x_batch,y_batch=data.train.next_batch(batch_size)

    # print(x_batch.shape,y_batch.shape)
    # print(y_batch)



    # print("shale of test image and labels", data.test.images.shape, data.test.labels.shape)

    _,loss=sess.run([train_step,cross_entropy],feed_dict={x: x_batch,y_: y_batch})
    # Accuracy = sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})
    if(i%500==0):
        print(i)
        # print("ifffffff")
        # print("shale of test image and labels",data.test.images.shape,data.test.labels.shape)
        Accuracy=sess.run(accuracy,feed_dict={x:data.test.images,y_: data.test.labels})

        # print("after ifffffffffff")
        Accuracy=round(Accuracy*100,2)
    #
        print ("Loss:{},Accuracy on test set :{} ".format(loss,Accuracy))
    #
    # elif(i%100==0):
    #     print ("Loss:{}".format(loss))

print('hhhhhhhhhhh')
'''
validation_accuracy=round((sess.run(accuracy,feed_dict={x: data.validation.images,y_:data.validation.labels}))*100,2)

print ("Accuracy in yhe validation data set :{}".format(validation_accuracy))


# test pridction

y_test=(sess.run(y_predicted,feed_dict={x:data.test.images}))


# confusion metrix

true_class=np.argmax(data.test.labels,1)
prediction_class=np.argmax(y_test,1)
cm=confusion_matrix(prediction_class,true_class)
cm
# print (cm)

# ploting confusion metrix

plt.imshow(cm,interpolation='nearest')
plt.colorbar()
numbber_of_classes=len(np.unique(true_class))
tick_marks=np.arange(len(np.unique(true_class)))
plt.xticks(tick_marks,range(numbber_of_classes))
plt.yticks(tick_marks,range(numbber_of_classes))
plt.tight_layout()
plt.ylabel('true_label')
plt.xlabel('predicted_label')
plt.title('confusion metrix')
plt.show()

# finding error output
idx=np.argmax(y_test,1)==np.argmax(data.test.labels,1)

# indicats the error output
cmp=np.where(idx==False)

# ploting error
fig,axes=plt.subplot(5,3,figsize=(15,15))
fig.subplots_adjust(hspace=0.3,wspace=0.3)
cls_true=np.argmax(data.test.labels,1)[cmp]
cls_pred=np.argmax(y_predicted,1)[cmp]
imags=data.test.images[cmp]
for i, ax in enumerate(axes.flat):
    ax.imshoW(imags[i].reshape(28,28),camp='binary')
    xlabel='True:{0}, Pred:{1}'.format(cls_true[i],cls_pred[i])
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


conv_layer1_filter=conv_layer.W.eval()
print (conv_layer1_filter.shape)

conv_layer1_filter_image=conv_layer1_filter[:,:,0,:]
print (conv_layer1_filter_image.shape)
# plotting filters
fig,axes=plt.subplots(8,4,figsize=(15,15))
fig.subplots_adjust(hspace=0.3,wspace=0.3)

for i, ax in enumerate(axes.flat):
    ax.imshoW(conv_layer1_filter_image[:,:,i],camp='gray')
    xlabel='filter: '.format(i+1)
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

test_image=np.reshape(data.test.images[0],[1,784])
conv_layer1_output=(sess.run(out_layer_1,feed_dict={x: test_image}))


plt.imshow(np.reshape(data.test.images[0],[28,28]),cmap='gray')
plt.title('Test image')
plt.axis('off')
plt.show()


print (conv_layer1_output.shape)

conv_layer_1_output_image=conv_layer1_output[0,:,:,:]
fig,axes=plt.subplots(8,4,figsize=(15,15))
fig.subplots_adjust(hspace=0.3,wspace=0.3)

for i, ax in enumerate(axes.flat):
    ax.imshoW(conv_layer1_filter_image[:,:,i],camp='gray')
    xlabel='filter: '.format(i+1)
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()




# testing on own handwritten data BLACK BACK BACKGROUND

im=Image.open("path_of_file")
im

im=im.resize((28,28),Image.ANTIALIAS)
im=np.array(im)
im2=im/np.max(im).astype(float)

test_image_1=np.reshape(im2,[1,784])

pred=(sess.run(y_predicted,feed_dict={x:test_image_1}))

prediction_class=np.argmax(pred)

print ("predicted class:{}".format(prediction_class))




print('hhhhhhhhhhh')

'''