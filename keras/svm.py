# SVM Starts from :111

from keras.datasets import cifar10
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train.astype('float64'), (X_train.shape[0], -1))
X_val = np.reshape(X_val.astype('float64'), (X_val.shape[0], -1))
X_test = np.reshape(X_test.astype('float64'), (X_test.shape[0], -1))
X_dev = np.reshape(X_dev.astype('float64'), (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
plt.show()

# second: subtract the mean image
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

learning_rates = [1e-7, 5e-5, 1]
regularization_strengths = [2.5e4, 5e4, 0.01]

num_epoch = 4

best = 1
best_r = 1
best_lr = 1

for r in regularization_strengths:
    for lr in learning_rates:

        opt = optimizers.SGD(learning_rate=lr)
        svm = Sequential()
        svm.add(Dense(64, activation='relu'))
        svm.add(Dense(1, kernel_regularizer=l2(r), activation='softmax'))
        svm.compile(loss='squared_hinge',
                        optimizer=opt,
                        metrics=['accuracy'])
        history = svm.fit(X_train, y_train, 
                            epochs = num_epoch,
                            batch_size = 64)

        print("Learning Rate: {}  Regularization: {}".format(lr,r))
        y_train_pred = svm.predict(X_train)
        print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
        y_val_pred = svm.predict(X_val)
        print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
        loss_val = np.mean(y_val == y_val_pred)
        if(best > loss_val and loss_val != 0):
            best = loss_val
            best_r = r
            best_lr = lr

print("Best LR: {}  R: {}".format(best_lr, best_r))
opt = optimizers.SGD(learning_rate=best_lr)
svm = Sequential()
svm.add(Dense(64, activation='relu'))
svm.add(Dense(1, kernel_regularizer=l2(best_r), activation='softmax'))
svm.compile(loss='squared_hinge',
                optimizer=opt,
                metrics=['accuracy'])
history = svm.fit(X_train, y_train, 
                    epochs = num_epoch,
                    batch_size = 64)

y_test_pred = svm.predict(X_test)
print('\nTest accuracy: %f' % (np.mean(y_test == y_test_pred), ))

train_loss = history.history['loss']

print(history.history.keys())

xc = range(num_epoch)
plt.figure()
train, = plt.plot(xc, train_loss, label='Train')
plt.legend()
plt.show()
