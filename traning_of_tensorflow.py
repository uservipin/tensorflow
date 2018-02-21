import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# print('hhhhhhhhiiiiiiiiiiiii11111111')
mnist= input_data.read_data_sets("/tmp/data/",one_hot=True)
# print('hhhhhhhhiiiiiiiiiiiii')

# hidden layer
n_nodes_hl_1= 5000
n_nodes_hl_2= 4000
n_nodes_hl_3= 1000


# no.  of classes
n_classes =10

# data is sent threw batch (no. of features sent threw at one time in ram)
# here 100 features sent in programm
batch_size= 100

# define placeholder
x= tf.placeholder('float',[None,784])
y= tf.placeholder('float')

def neural_network_model(data):
    print('hhhhhhhhiiiiiiiiiiiii')

    hidden_1_layer ={'weight':tf.Variable(tf.random_normal([784,n_nodes_hl_1])),
                     'baises':tf.Variable(tf.random_normal([n_nodes_hl_1]))}


    hidden_2_layer ={'weight':tf.Variable(tf.random_normal([n_nodes_hl_1,n_nodes_hl_2])),
                     'baises':tf.Variable(tf.random_normal([n_nodes_hl_2]))}

    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl_2, n_nodes_hl_3])),
                      'baises': tf.Variable(tf.random_normal([n_nodes_hl_3]))}

    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl_3, n_classes])),
                      'baises': tf.Variable(tf.random_normal([n_classes]))}


    l1= tf.add(tf.matmul(data,hidden_1_layer['weight']),hidden_1_layer['baises'])
    l1 =tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['baises'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['baises'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['baises']

    return  output


def train_neural_etwork(x):

    prediction =neural_network_model(x)

    squared_data = tf.square(prediction - y)
    cost = tf.reduce_sum(squared_data)

    # cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)

    hm_epochs =10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())


        # doubts ...............for epoc in range(hm_epochs:
        for epoch in range(hm_epochs):
            epoch_loss =0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                _,c= sess.run([optimizer,cost],feed_dict= {x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('epoch',epoch,'completed out of ', hm_epochs,'loss:',epoch_loss)

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


train_neural_etwork(x)




















