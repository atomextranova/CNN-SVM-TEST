import keras
import os
import sys
import numpy as np
import logging
import keras.backend as K
from keras.datasets import cifar10
import h5py
from keras.optimizers import Adam


def KerasModelWrapper(_model):

    def __init__(self, _model, softmax=False):
        if softmax:
            self.model = clip_model(_model)
        else:
            self.model = _model
        self.make_gradient_function()

    def make_gradient_function(self):
        input = self.model.input
        backend_support = ['tensorflow']
        if K.backend() in backend_support:
            if K.backend() == 'tensorflow':
                logit_value = model.outputs

        else:
            raise NotImplementedError

    def gradient_wrt_output(self, input_image, label):
        pass

    def gradient_wrt_loss:
        pass

def deep_fool(model, input_images, label, size_index, p_norm, num_classes = 10, subsample=0, max_iteration=1000):
    if input_images.shape[size_index] != label.shape[size_index]:
        raise ValueError("The amount of images is not equal to that of labels")

    num_classes = 10

    if subsample != 0:
        if num_classes <= subsample:
            logging.warning(
                "Use real class number {} instead".format(
                    num_classes))
            subsample = num_classes - 1
        else:
            logging.info('Only Using the top-{} classes for perturbation'.format(subsample))
            new_label = np.argsort(label, axis=1)[:, 0:subsample]
    else:
        logging.info("Using the all {} labels for adversarial perturbation".format(num_classes))

    grad = K.gradients(model.output, model.input)

    # with K.get_session() as sess:
    #     gradient = sess.run(grad, feed_dict={model.input: input_image})
    with K.get_session() as sess:
        for image in input_images:
            for step in range(max_iteration):
                pass




    # sign = np.sign(gradient)
    print(grad)

def clip_model(model):
    # model.layers.pop()
    # x = model.layers[-1].output
    # model1 = keras.Model(input=model.input, output=x)
    # model1.compile(loss='categorical_crossentropy',
    #                optimizer='adam',
    #                metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = keras.models.load_model("attack/cifar10_ResNetSVM20v3_model.156.35.0.001.h5")
    # with h5py.File("attack/orig.h5") as file:
    #     image = file['orig'][::10]
    #     label = file['label'][::10]

    subtract_pixel_mean = True

    num_classes = 10
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

    # clip_model = clip_model(model)
    print(model.summary())
    deep_fool(model, x_test[::10], y_test[::10], 0, 2, subsample=5)
