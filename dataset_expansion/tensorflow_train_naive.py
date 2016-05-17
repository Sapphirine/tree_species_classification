
import csv
DATAPATH="../leafsnap2-dataset/dataset/"
import tensorflow as tf
import numpy as np
import random
import math
import os
from itertools import cycle
from scipy import misc

BATCH_SIZE = 1

LEARNING_RATE= 0.001
TRAINING_ITER= 1000000
DISPLAY_STEP= 20
DROPOUT = 0.75
SAVE_ITER = 100
IMAGE_SIZE = 300
NUM_CLASSES = 185


weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32]), name="wc1"),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="wc2"),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([75*75*64, 1024]), name="wd1"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]), name="out_weight")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name="bc1"),
    'bc2': tf.Variable(tf.random_normal([64]), name="bc2"),
    'bd1': tf.Variable(tf.random_normal([1024]), name="bd1"),
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]), name="out_bias")
}


###Binomial distribution on the number of labels
#sigmoidal n from 0. to 1
#s = np.random.binomial(100, n , allspecies) ##determines the number or percent of samples in the species to take from lab vs field
#once convergence apply arbitrary mutations translations rotations
#rand betwene 0 and 360 rotation
#rand between 0 100 fo gaussian noise filter
def onehotvectorize(i):
    vc = np.zeros(NUM_CLASSES)
    vc[i] = 1
    return vc



class ImageLabelIterator:
    def __init__(self, allfiles, labelsdict):
        self.allfiles = allfiles
        self.labelsdict = labelsdict
        self.labelsiter = cycle(labelsdict.keys())
        self.i = 0

    def __iter__(self):
        return self

    def next(self, mutationrate, mutation, proboffield):
        numoffiled = list(np.random.binomial(1, proboffield, BATCH_SIZE))
        returnimages = []
        returnlabels = []
        while self.i < BATCH_SIZE:
            label = self.labelsiter.next()
            if numoffiled[self.i] is 1 and (label, 'field') in numoffiled:
                if np.random.binomial(1, mutationrate, 1)[0] == 1:
                    returnimages.append(mutation(random.choice(self.allfiles[(label, 'field')])))
                    returnlabels.append(onehotvectorize(self.labelsdict[label]))
                else:
                    returnimages.append(misc.imresize(misc.imread(random.choice(self.allfiles[(label, 'field')])), (IMAGE_SIZE, IMAGE_SIZE, 3)))
                    returnlabels.append(onehotvectorize(self.labelsdict[label]))
            else:
                if np.random.binomial(1, mutationrate, 1)[0] == 1:
                    returnimages.append(mutation(random.choice(self.allfiles[(label, 'lab')])))
                    returnlabels.append(onehotvectorize(self.labelsdict[label]))
                else:
                    returnimages.append(misc.imresize(misc.imread(random.choice(self.allfiles[(label, 'lab')])), (IMAGE_SIZE, IMAGE_SIZE, 3)))
                    returnlabels.append(onehotvectorize(self.labelsdict[label]))
            self.i+=1
        self.i = 0
        return returnimages, returnlabels


def image_label_generator(allfiles, labelsdict):
    imagebatch = []
    labelbatch = []
    for k, t in enumerate(cycle(allfiles.keys())):
        if k % BATCH_SIZE == 0:
            yield imagebatch, labelbatch
            imagebatch = []
            labelbatch = []
        specie, laborfield = t
        yield misc.imresize(misc.imread(random.choice(allfiles[(specie, laborfield)])), (IMAGE_SIZE, IMAGE_SIZE, 3)), onehotvectorize(labelsdict[specie])




def genratebatch(allfiles,lablesdict, fieldprobability, mutation, mutationrate):
    imagebatch = []
    labelbatch = []
    allspecies = allfiles.keys()

    numoffiled = list(np.random.binomial(BATCH_SIZE, fieldprobability, len(allspecies)))
    for k, t in enumerate(allspecies):
            specie, laborfield = t
            for i in xrange(numoffiled[k]):
                labfield = "field"
                if (specie, "field") not in allfiles:
                    labfield = "lab"
                if np.random.binomial(1, mutationrate, 1)[0] == 1:
                    imagebatch.append(mutation(misc.imresize(misc.imread(random.choice(allfiles[(specie, labfield)])), (IMAGE_SIZE, IMAGE_SIZE, 3))))
                    labelbatch.append(onehotvectorize(lablesdict[specie]))
                else:
                    imagebatch.append(misc.imresize(misc.imread(random.choice(allfiles[(specie, labfield)])), (IMAGE_SIZE, IMAGE_SIZE, 3)))
                    labelbatch.append(onehotvectorize(lablesdict[specie]))
            for l in xrange(BATCH_SIZE - numoffiled[k]):
                labfield = "lab"
                if (specie, "lab") not in allfiles:
                    labfield = "field"
                if np.random.binomial(1, mutationrate, 1)[0] == 1:
                    imagebatch.append(mutation(misc.imresize(misc.imread(random.choice(allfiles[(specie, labfield)])), (IMAGE_SIZE, IMAGE_SIZE, 3))))
                    labelbatch.append(onehotvectorize(lablesdict[specie]))
                else:
                    imagebatch.append(misc.imresize(misc.imread(random.choice(allfiles[(specie, labfield)])), (IMAGE_SIZE, IMAGE_SIZE, 3)))
                    labelbatch.append(onehotvectorize(lablesdict[specie]))

    return imagebatch, labelbatch


def mutation(img):
    return img




def generate_allfiles(file):
    with open(file,'rb') as tsvin:
        # l = tsvin.readline()
        # fieldnames = l.split("\t")
        # fieldnames.append(fieldnames[1].rstrip("\n"))

        allfiles = dict()
        lablesdict = set()
        reader = csv.reader(tsvin.readlines(), delimiter='\t')
        for i, line in enumerate(reader):
            if i > 0:
                lablesdict.add(line[-2])
                allfiles.setdefault((line[-2],line[-1]), []).append(line[1])
    lablesdict = dict(zip(list(lablesdict), range(len(lablesdict))))
    return allfiles, lablesdict



def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1],
                                                  padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 300, 300, 3])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    # Relu activation
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
    # Apply Dropout
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out


def main(file, modelpath):

    allfiles, lablesdict = generate_allfiles(file)
    ##TF vars
    dropout = 0.75
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="x")
    print len(lablesdict)
    y = tf.placeholder(tf.float32, shape=[None, len(lablesdict)], name="y")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") #dropout (keep probability)


    #Model construction
    pred = conv_net(x, weights, biases, keep_prob)

    #LOSS AND OPTIMIZER
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name="optimizer").minimize(cost)

    #EVAL MODEL
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Initializing the variables
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    #inputiterator = ImageLabelIterator(allfiles, lablesdict)
    #inputiterator = image_label_generator(allfiles, lablesdict)
    with tf.Session() as sess:
        sess.run(init)

        if os.listdir(modelpath) != []:
            saver.restore(sess, modelpath+"/model.ckpt")
            print "loaded"
        i = 0
        while i * BATCH_SIZE < TRAINING_ITER:

            proboffield = 0.5*(1 + math.sin(2*3.14*.1* i))
            mutationrate = 0.5*(1 + math.sin(2*3.14*.01* i))
            imagebatch, labelbatch = genratebatch(allfiles, lablesdict, proboffield, mutation, mutationrate)
            #imagebatch, labelbatch = inputiterator.next(mutationrate, mutation, proboffield)

            sess.run(optimizer, feed_dict={x: imagebatch, y: labelbatch, keep_prob: dropout})

            if i % DISPLAY_STEP == 0:
                acc = sess.run(accuracy, feed_dict={x:imagebatch, y:labelbatch, keep_prob:1.})
                loss = sess.run(cost, feed_dict={x:imagebatch, y:labelbatch, keep_prob:1.})
                print "Iter " + str(i * BATCH_SIZE) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

            if i % SAVE_ITER == 0:
                print "saved"
                saver.save(sess, modelpath+"/model.ckpt")

            i+=1






if __name__ == "__main__":
    file="leafsnap2-dataset-images.txt"
    modelpath = "model"
    main(file, modelpath)