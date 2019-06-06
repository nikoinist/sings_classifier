
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from random import randint
import matplotlib.pyplot as plt
# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']



```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below.


```python


### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_test[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))



print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
    

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.


```python
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
```


```python
import random

%matplotlib inline

fig, axes = plt.subplots(nrows=3, ncols=3)
fig.subplots_adjust(hspace=0.3, wspace=0.5)

for i,ax in enumerate(axes.flat):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    ax.imshow(image.squeeze())
    ax.set_title("label: {0}".format(y_train[index]))
    ax.set_xticks([])
    ax.set_yticks([])
```


![png](output_7_0.png)



```python
plt.hist(y_train, bins=n_classes)
```




    (array([  210.,  2220.,  2250.,  1410.,  1980.,  1860.,   420.,  1440.,
             1410.,  1470.,  2010.,  1320.,  2100.,  2160.,   780.,   630.,
              420.,  1110.,  1200.,   210.,   360.,   330.,   390.,   510.,
              270.,  1500.,   600.,   240.,   540.,   270.,   450.,   780.,
              240.,   689.,   420.,  1200.,   390.,   210.,  2070.,   300.,
              360.,   240.,   240.]),
     array([  0.        ,   0.97674419,   1.95348837,   2.93023256,
              3.90697674,   4.88372093,   5.86046512,   6.8372093 ,
              7.81395349,   8.79069767,   9.76744186,  10.74418605,
             11.72093023,  12.69767442,  13.6744186 ,  14.65116279,
             15.62790698,  16.60465116,  17.58139535,  18.55813953,
             19.53488372,  20.51162791,  21.48837209,  22.46511628,
             23.44186047,  24.41860465,  25.39534884,  26.37209302,
             27.34883721,  28.3255814 ,  29.30232558,  30.27906977,
             31.25581395,  32.23255814,  33.20930233,  34.18604651,
             35.1627907 ,  36.13953488,  37.11627907,  38.09302326,
             39.06976744,  40.04651163,  41.02325581,  42.        ]),
     <a list of 43 Patch objects>)




![png](output_8_1.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

There are various aspects to consider when thinking about this problem:

- Neural network architecture
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

**NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!


```python

```

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
#Randomize the training data
X_train, y_train = shuffle(X_train, y_train)


```

### Question 1 

_Describe how you preprocessed the data. Why did you choose that technique?_

**Answer:**
I used the shuffle function inside the sklearn toolkit to randomize the train data. So the probability of all the signs in the data are used.


```python
#Splitting the train data into training and validation data.
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=101)
### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

```

### Question 2

_Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

**Answer:**
I used the train_test_split function from the sklearn package to split the train data into 80/20 ratio for training and validation. 


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.

#paramaters and variables
rate = 0.00020
EPOCHS = 20
BATCH_SIZE = 150
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)
keep_probability = 1.
stddev=0.01

#racunanje convolucija ulaz_visina - conv_visina + 1)/1
#pooling ulaz_visina - con_visina + 2*P)/S+1

#layers weights & biases
weights = {
    # 5x5 conv, 3 input, 64 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=stddev)),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=stddev)),
    # fully connected, 8*8*64 inputs, 1300 outputs
    'wd1': tf.Variable(tf.truncated_normal([8*8*64, 1300], stddev=stddev)),
    # fully connected, 1300, 1000
    "wd2": tf.Variable(tf.truncated_normal([1300, 1000], stddev=stddev)),
    # out 1000 inputs, 43 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1000, n_classes], stddev=stddev))
}

biases = {
    'bc1': tf.Variable(tf.zeros([64])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bd1': tf.Variable(tf.zeros([1300])),
    'bd2': tf.Variable(tf.zeros([1000])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

#Functions for convolution, pooling & fully connected
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    #bias
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def fully_conect(x, W, b, use_relu=True): 
    layer = tf.matmul(x, W) + b
    
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer

#creating model
def conv_nn(x, weights, biases, keep_prob):
    
    #Convolution layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #Convolution layer 1 :maxpooling
    conv1 = maxpool2d(conv1, k=2)
    #Convolution layer 1 :dropout
    conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob) 
    
    #Convolution layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #Convolution layer 2 :maxpooling
    conv2 = maxpool2d(conv2, k=2)
    #Convolution layer 2 :dropout
    conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)
    
    #Flatten layer
    conv2 = flatten(conv2)
    
    #Fully Connected 1
    fully1 = fully_conect(conv2, weights['wd1'], biases['bd1'], use_relu=True)
    #Fully Connected 1: dropout
    fully1 = tf.nn.dropout(fully1, keep_prob=keep_prob)
    
    #Fully Connected 2
    fully2 = fully_conect(fully1, weights['wd2'], biases['bd2'], use_relu=True)
    #Fully Connected 2: dropout
    fully2 = tf.nn.dropout(fully2, keep_prob=keep_prob)
    
    #output
    logits = fully_conect(fully2, weights['out'], biases['out'], use_relu=False)
    
    return logits

#construct model
logits= conv_nn(x, weights, biases, keep_prob)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate).minimize(loss_operation)
#training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
```

### Question 3

_What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._


**Answer:**
I used the simple LeNet for my architecture. There are 64 filters per convolusional layer. All the hidden layers have a dropout applied on them and each layer feed into the next.
- Convolution
- Pooling
- Dropout
- Convolution
- Pooling
- Dropout
- Fully connected
- Dropout
- Fully connected
- Output



```python


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        #X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:keep_probability})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, 'traffic')
    print("Model saved")
### Train your model here.
### Feel free to use as many code cells as needed.
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.848
    
    EPOCH 2 ...
    Validation Accuracy = 0.964
    
    EPOCH 3 ...
    Validation Accuracy = 0.974
    
    EPOCH 4 ...
    Validation Accuracy = 0.986
    
    EPOCH 5 ...
    Validation Accuracy = 0.983
    
    EPOCH 6 ...
    Validation Accuracy = 0.987
    
    EPOCH 7 ...
    Validation Accuracy = 0.986
    
    EPOCH 8 ...
    Validation Accuracy = 0.967
    
    EPOCH 9 ...
    Validation Accuracy = 0.989
    
    EPOCH 10 ...
    Validation Accuracy = 0.991
    
    EPOCH 11 ...
    Validation Accuracy = 0.985
    
    EPOCH 12 ...
    Validation Accuracy = 0.994
    
    EPOCH 13 ...
    Validation Accuracy = 0.989
    
    EPOCH 14 ...
    Validation Accuracy = 0.993
    
    EPOCH 15 ...
    Validation Accuracy = 0.989
    
    EPOCH 16 ...
    Validation Accuracy = 0.990
    
    EPOCH 17 ...
    Validation Accuracy = 0.990
    
    EPOCH 18 ...
    Validation Accuracy = 0.986
    
    EPOCH 19 ...
    Validation Accuracy = 0.992
    
    EPOCH 20 ...
    Validation Accuracy = 0.995
    
    Model saved
    


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.953
    

### Question 4

_How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_


**Answer:**
I used the Adam optimizer witch works a lot better and faster than the gradient descent optimizer. I started with the larger batch of about 500 but lovered it to 150 since it worked quite well. I set the epochs to 20 witch gives the best results.Also lowered the training rate from the initial from the LeNet and found it to be better

### Question 5


_What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

**Answer:**
I started as instructed with LeNet from the lab excercise but found it to only train model to about 60% accuracy. After looking around tutorials and lessons for network training code I found that they use less fully connected layers and removed one from the LeNet style network witch worked fine.
Looking at the weight and biases I doubled the numbers and lovered the training rate for more accuracy, also added the deviation for better fitting.

---

## Step 3: Test a Model on New Images

Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
my_img_file = "new_signs.p"

with open(my_img_file, mode='rb') as f:
    my_test = pickle.load(f)
    
XX_test, yy_test = my_test['features'], my_test['labels']

```


```python

```

### Question 6

_Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._



**Answer:** There is a 30 km/h limit sign that is specific for my location and has a some spray paint over it with a name in local language. And some of the signs are slightly askew.


```python
### Run the predictions here.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy= evaluate(XX_test, yy_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Test Accuracy = 0.800
    

### Question 7

_Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._

_**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._


**Answer:**
My model performed well on my test images with 80% accuracy.


```python
softmax = tf.nn.softmax(logits)
top_5_op = tf.nn.top_k(softmax,5)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_5 = sess.run(top_5_op, feed_dict={x: XX_test, y: yy_test, keep_prob:1})
    print(top_5)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,5))
fig.subplots_adjust(hspace=0.3, wspace=0.3)


for i,ax in enumerate(axes.flat):
    ax.imshow(XX_test[i].squeeze())
    ax.set_title("label: {0}".format(yy_test[i]))
    ax.set_xlabel("prediction: {0}".format(top_5[1][i][0]))
    ax.set_xticks([])
    ax.set_yticks([])    


### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
```

    TopKV2(values=array([[  9.95417833e-01,   4.58100252e-03,   1.21065921e-06,
              4.65885002e-08,   2.55799470e-09],
           [  9.99971509e-01,   2.83737827e-05,   7.84946650e-08,
              1.98776249e-08,   3.68354541e-10],
           [  9.02846634e-01,   6.87898621e-02,   2.77455505e-02,
              4.94960463e-04,   1.14082293e-04],
           [  9.99933124e-01,   5.98751103e-05,   6.01611009e-06,
              9.44559361e-07,   2.69434631e-11],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00],
           [  7.72284806e-01,   6.61096722e-02,   4.13126387e-02,
              3.88018191e-02,   2.20044591e-02],
           [  1.00000000e+00,   4.85606947e-25,   1.30250895e-25,
              1.31649006e-26,   5.21479716e-31],
           [  1.00000000e+00,   3.58050949e-15,   2.67123007e-17,
              3.12658822e-22,   7.83506369e-30],
           [  1.00000000e+00,   1.67995739e-11,   1.01828257e-12,
              3.11711547e-14,   1.75675006e-14],
           [  9.98634636e-01,   9.72051697e-04,   3.89172899e-04,
              3.22820915e-06,   7.49274818e-07]], dtype=float32), indices=array([[ 9, 41, 23, 36, 19],
           [13, 35,  9,  3, 15],
           [14, 25, 12, 13, 29],
           [ 1,  2,  6, 21, 31],
           [38,  0,  1,  2,  3],
           [38,  3, 29, 22, 34],
           [17, 14,  0,  8, 26],
           [18, 27, 24, 26, 29],
           [28, 29, 20, 30, 24],
           [14, 15,  8, 17, 29]], dtype=int32))
    Tensor("strided_slice:0", shape=(43,), dtype=float32)
    


![png](output_36_1.png)



```python

```


```python
import math
for index in range(len(top_5[0])):

    plt.bar(top_5[1][index], top_5[0][index])
    plt.ylim(-.01, 0.1)
    plt.xlabel("Classes")
    plt.ylabel("Softmax")
    plt.show()
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)



![png](output_38_4.png)



![png](output_38_5.png)



![png](output_38_6.png)



![png](output_38_7.png)



![png](output_38_8.png)



![png](output_38_9.png)


### Question 8

*Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python

```

**Answer:**
Looking at the top k most certain is of the clean front facing signs. But for most false predictions the real label is in top 5 k's except the spectial limit sign with some paint on the sign and local name for a zone.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.


```python

```
