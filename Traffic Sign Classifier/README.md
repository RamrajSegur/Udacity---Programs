
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
import numpy as np
import matplotlib.pyplot as plt
n_train = len(X_train)

# TODO: Number of validation examples
n_validation =len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[1].shape

# TODO: How many unique classes/labels there are in the dataset.
unique=np.unique(y_train)
# print (unique)
un,indices=np.unique(y_train,return_index=True)
n_classes = len(unique)

con=np.empty((43))
list_y=list(y_train)
for i in range(43):
    con[i]=(list_y.count(i))
    
y_pos = np.arange(0,43)
performance = con
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xlabel('Image Index')
plt.ylabel('Number of Images')
plt.title('Image Data Distribtuion')
plt.show()

# print (n_validation)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```


![png](README_files/README_5_0.png)


    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
    

### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
# Visualizations will be shown in the notebook.
%matplotlib inline
x_train=np.empty((34799,21,21,3))
index = 21
imgs=np.empty((32,32,3))
image = X_train[index]
print (image.shape)
plt.imshow(image)
imgs1=np.zeros((21,21,3))
imgs1=image[4:25,4:25]

plt.figure(figsize=(1,1))
# plt.imshow(imgs)
plt.imshow(imgs1)
print (imgs1.shape)
print(y_train[index])
```

    (32, 32, 3)
    (21, 21, 3)
    41
    


![png](README_files/README_8_1.png)



![png](README_files/README_8_2.png)



```python
fig, axs = plt.subplots(9,5,figsize=(15, 30))
print (indices)
axs = axs.ravel()
for i in range(len(indices)):
    image = X_train[indices[i]]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[indices[i]])

```

    [ 9960  2220 31439  5370  6810 12360 21450 23730 15870 11040 17130  8580
     27329 21810 29219 29909  5010 30449 20370  6630 25950 25680  4500  1770
     10800 33449  1230 10350 26849 10560 25020   210 10140 26250 20010 18930
       900  4830 14010 25410  4200     0  9750]
    


![png](README_files/README_9_1.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2
x_train=[]
x_valid=[]
%matplotlib inline
for i in range(len(X_train)):
    img=np.empty((21,21,3))
    image=X_train[i,7:28,5:26]
#     plt.imshow(image) 
    image1=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.addWeighted(image,1.5,image1,-0.5,0)
    x_train.append(image)
for i in range(len(X_valid)):
    img=np.empty((21,21,3))
    image=X_valid[i,7:28,5:26]
    image1=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.addWeighted(image,1.5,image1,-0.5,0)
    x_valid.append(image)


# plt.imshow(X_train[30])
plt.imshow(x_train[21])




```




    <matplotlib.image.AxesImage at 0x26fa197a9b0>




![png](README_files/README_13_1.png)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.05
    keep_prob=0.9
    
    # Layer 1: Convolutional. Input = 21x21x1. Output = 18x18x20.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 3, 20), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(20))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    print (conv1)

    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    # Dropout 1
    conv1 = tf.nn.dropout(conv1, keep_prob)
    
    # Pooling. Input = 18x18x20. Output = 9x9x20.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    print (conv1)

    # Layer 2: Convolutional. Output = 6x6x40.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 20, 40), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(40))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    print (conv2)
    
    # Activation.
    conv2 = tf.nn.relu(conv2)    
    
    # Dropout 2
    conv1 = tf.nn.dropout(conv2, keep_prob)

    # Pooling. Input = 6x6x40. Output =3x3x40.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print (conv2)

    # Flatten. Input = 3x3x40. Output = 360.
    fc0   = flatten(conv2)
    print (fc0)
    
   
    # Fully Connected. Input = 360. Output = 200.
    fcbb_W = tf.Variable(tf.truncated_normal(shape=(360, 200), mean = mu, stddev = sigma))
    fcbb_b = tf.Variable(tf.zeros(200))
    fcbb   = tf.matmul(fc0, fcbb_W) + fcbb_b
    
    # Activation.
    fcbb    = tf.nn.relu(fcbb)
    
    # Layer 2: Fully Connected. Input = 200. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(200, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fcbb, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    
    # Layer 3: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 100), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(100))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    
    # Layer 4: Fully Connected. Input = 100. Output = 80.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(100, 80), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(80))
    fc4    = tf.matmul(fc2, fc4_W) + fc4_b
    
    # Activation.
    fc4    = tf.nn.relu(fc4)
    
    # Layer 5: Fully Connected. Input = 80. Output = 60.
    fc5_W  = tf.Variable(tf.truncated_normal(shape=(80, 60), mean = mu, stddev = sigma))
    fc5_b  = tf.Variable(tf.zeros(60))
    fc5    = tf.matmul(fc4, fc5_W) + fc5_b
    
    # Activation.
    fc5    = tf.nn.relu(fc5)

    # Layer 6: Fully Connected. Input = 60. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(60,43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc5, fc3_W) + fc3_b
    
    return logits
```


```python
import tensorflow as tf
x = tf.placeholder(tf.float32, (None, 21, 21, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
```


```python
rate = 0.0005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

    Tensor("add:0", shape=(?, 18, 18, 20), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 9, 9, 20), dtype=float32)
    Tensor("add_1:0", shape=(?, 6, 6, 40), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 3, 3, 40), dtype=float32)
    Tensor("Flatten/Reshape:0", shape=(?, 360), dtype=float32)
    


```python
import tensorflow as tf

EPOCHS = 35
BATCH_SIZE = 128
```


```python
from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train)


```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
a=tf.nn.softmax(logits,dim=-1,name=None)
[data,b]=tf.nn.top_k(a,k=5,sorted=True,name=None)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
             
        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.733
    
    EPOCH 2 ...
    Validation Accuracy = 0.892
    
    EPOCH 3 ...
    Validation Accuracy = 0.915
    
    EPOCH 4 ...
    Validation Accuracy = 0.930
    
    EPOCH 5 ...
    Validation Accuracy = 0.937
    
    EPOCH 6 ...
    Validation Accuracy = 0.934
    
    EPOCH 7 ...
    Validation Accuracy = 0.935
    
    EPOCH 8 ...
    Validation Accuracy = 0.954
    
    EPOCH 9 ...
    Validation Accuracy = 0.954
    
    EPOCH 10 ...
    Validation Accuracy = 0.954
    
    EPOCH 11 ...
    Validation Accuracy = 0.953
    
    EPOCH 12 ...
    Validation Accuracy = 0.939
    
    EPOCH 13 ...
    Validation Accuracy = 0.951
    
    EPOCH 14 ...
    Validation Accuracy = 0.964
    
    EPOCH 15 ...
    Validation Accuracy = 0.956
    
    EPOCH 16 ...
    Validation Accuracy = 0.958
    
    EPOCH 17 ...
    Validation Accuracy = 0.955
    
    EPOCH 18 ...
    Validation Accuracy = 0.966
    
    EPOCH 19 ...
    Validation Accuracy = 0.955
    
    EPOCH 20 ...
    Validation Accuracy = 0.966
    
    EPOCH 21 ...
    Validation Accuracy = 0.965
    
    EPOCH 22 ...
    Validation Accuracy = 0.958
    
    EPOCH 23 ...
    Validation Accuracy = 0.961
    
    EPOCH 24 ...
    Validation Accuracy = 0.960
    
    EPOCH 25 ...
    Validation Accuracy = 0.963
    
    EPOCH 26 ...
    Validation Accuracy = 0.958
    
    EPOCH 27 ...
    Validation Accuracy = 0.954
    
    EPOCH 28 ...
    Validation Accuracy = 0.967
    
    EPOCH 29 ...
    Validation Accuracy = 0.960
    
    EPOCH 30 ...
    Validation Accuracy = 0.967
    
    EPOCH 31 ...
    Validation Accuracy = 0.949
    
    EPOCH 32 ...
    Validation Accuracy = 0.965
    
    EPOCH 33 ...
    Validation Accuracy = 0.969
    
    EPOCH 34 ...
    Validation Accuracy = 0.958
    
    EPOCH 35 ...
    Validation Accuracy = 0.970
    
    Model saved
    


```python
# batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
x_test=[]
for i in range(len(X_test)):
    img=np.empty((21,21,3))
    image=X_test[i,7:28,5:26]
#     plt.imshow(image) 
    image1=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.addWeighted(image,1.5,image1,-0.5,0)
    x_test.append(image)

with tf.Session() as sess:
    saver.restore(sess, './lenet')
    accuracy = sess.run(accuracy_operation, feed_dict={x: x_test, y: y_test})
    print ("Test Accuracy = {:.3f}".format(accuracy))

```

    Test Accuracy = 0.951
    

---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob

imaged = [cv2.imread(file) for file in glob.glob("C:/Users/ramra/CarND-Term1-Starter-Kit/CarND-Traffic-Sign-Classifier-Project-master/traffic-signs-data/custom/*.png")]

y_downloaded=[39,40,28,17,13]
for i in range(len(imaged)):
    imaged[i]=cv2.cvtColor(imaged[i], cv2.COLOR_BGR2RGB)
fig, axs = plt.subplots(1,5,figsize=(15, 3))
axs = axs.ravel()
for i in range(5):
    image = imaged[i]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_downloaded[i])

```


![png](README_files/README_27_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
x_imaged=[]
for i in range(len(imaged)):
    img=np.empty((21,21,3))
    imaged1=imaged[i]
    image=imaged1[7:28,5:26]
#     plt.imshow(image) 
    image1=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.addWeighted(image,1.5,image1,-0.5,0)
    x_imaged.append(image)

# for i in range(len(x_imaged)):
plt.imshow(x_imaged[4])

```




    <matplotlib.image.AxesImage at 0x26fa1df15c0>




![png](README_files/README_29_1.png)


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess, './lenet')
    result_index = sess.run(b, feed_dict={x: x_imaged})
    result_value = sess.run(data, feed_dict={x: x_imaged})
    accuracy=sess.run(accuracy_operation,feed_dict={x: x_imaged, y: y_downloaded})
    print (accuracy)

```

    1.0
    

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

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
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
print ((result_index))
print ((result_value))
```

    [[39 28 37 15  8]
     [40 38 39  2  5]
     [28 29 24 30 19]
     [17 26  0 29 33]
     [13 15 30 38 12]]
    [[  1.00000000e+00   1.79922351e-14   1.94531577e-15   1.88479590e-15
        7.68984949e-16]
     [  1.00000000e+00   4.32205554e-20   9.05061118e-28   1.21481405e-31
        6.44520834e-33]
     [  1.00000000e+00   1.38212246e-14   1.32726912e-14   9.81309375e-16
        1.35719063e-21]
     [  1.00000000e+00   3.35173625e-11   4.24050581e-16   5.63726565e-18
        3.11158928e-19]
     [  1.00000000e+00   2.48587872e-36   6.64831640e-37   3.59990302e-37
        1.64604455e-37]]
    

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
