from __future__ import print_function

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model


def resnet_ensemble(clip=False):
    input_layer = Input(input_shape)
    model_path = sys.argv[1]
    model_list = []
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('h5'):
                temp_model = keras.models.load_model(os.path.join(root, file))
                temp_model.name = file
                # for layer in temp_model.layers:
                #     layer.name = file + layer.name
                # new_output = None
                if "svm" not in file:
                    # temp_model.layers.pop()
                    # temp_output = temp_model.layers[-1].output
                    # temp_model_clipped = Model(inputs=temp_model.input, outputs=temp_output, name=file)
                    # print(temp_model_clipped.summary())
                    temp_model.layers[-1].activation = keras.layers.activations.linear
                    # keras.utils.apply_modifications(temp_model)
                    new_output = temp_model(input_layer)
                else:
                    new_output = temp_model(input_layer)
                # new_output.name = file + new_output.name
                # new_model = new_output
                # new_model = Model(inputs=input_layer, outputs=new_output, name=file)
                # print(new_model.summary)
                model_list.append(new_output)
    # output_tensors = [model.output for model in model_list]
    # final_output = keras.;layers.average(output_tensors)
    if len(model_list) != 1:
        final_output = keras.layers.average(model_list)
    else:
        final_output = model_list[0]
    ensemble_model = Model(inputs=input_layer, outputs=final_output, name='ensemble')
    print(ensemble_model.summary())
    return ensemble_model, 'Ensemble_ResNet{}'.format()


num_classes = 10

subtract_pixel_mean = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


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


model, model_type = resnet_ensemble()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
scores = model.evaluate(x_test, y_test, verbose=1)

file = open('result-%s.txt' % model_type, 'w')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

file.write('Test loss:' + str(scores[0]))
file.write('Test accuracy:' + str(scores[1]))

file.close()



model.save('ensemble.h5')

# print(model.predict(x_test[::100]))

#
# model.save_weights(model_type + '.hdh5')