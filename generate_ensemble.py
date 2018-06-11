from __future__ import print_function

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model


def resnet_ensemble(input_layer):
    model_path = sys.argv[1]
    model_list = []
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('h5'):
                temp_model = keras.models.load_model(os.path.join(root, file))
                temp_model.name = file
                temp_model.layers.pop(0)
                # for layer in temp_model.layers:
                #     layer.name = file + layer.name

                new_output = temp_model(input_layer)
                # new_output.name = file + new_output.name
                # new_model = new_output
                new_model = Model(inputs=input_layer, outputs=new_output, name=file)
                model_list.append(new_model)
    output_tensors = [model.output for model in model_list]
    final_output = keras.layers.average(output_tensors)
    ensemble_model = Model(inputs=input_layer, outputs=final_output, name='ensemble')
    print(ensemble_model.summary())
    return ensemble_model, 'Ensemble_ResNet%dv%d' % (depth, version)


depth = 20
version = 3
num_classes = 10

subtract_pixel_mean = True

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

new_input = Input(input_shape)
model, model_type = resnet_ensemble(new_input)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# scores = model.evaluate(x_test, y_test, verbose=1)

# file = open('result-%s.txt' % model_type, 'w')
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
#
# file.write('Test loss:' + str(scores[0]))
# file.write('Test accuracy:' + str(scores[1]))
#
# file.close()

print([lay.name for lay in model.input_layers])

#
# model.save_weights(model_type + '.hdh5')