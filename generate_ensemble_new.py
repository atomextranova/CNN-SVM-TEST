from __future__ import print_function

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
import numpy as np
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
import h5py
import argparse


def resnet_ensemble(model_locs, save_dir, image_shape):
    input_layer = Input(image_shape)
    model_list = []
    for root, _, files in os.walk(model_locs):
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
    return ensemble_model, 'Ensemble_ResNet%dv%d' % (depth, version)


def generate_orig():
    if not os.path.exists('orig.h5'):
        subtract_pixel_mean = True

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        x_train_mean = np.mean(x_train, axis=0)
        if subtract_pixel_mean:
            x_train -= x_train_mean
            x_test -= x_train_mean

        def array_to_scalar(arr):
            list = []
            for item in arr:
                list.append(np.asscalar(item))
            return np.array(list)

        # Convert class vectors to binary class matrices.
        y_train = array_to_scalar(y_train)
        y_test = array_to_scalar(y_test)

        # Save data
        with h5py.File("orig.h5", "w") as hf:
            hf.create_dataset(name='image', data=x_test)
            hf.create_dataset(name='label', data=y_test)
            hf.create_dataset(name='mean', data=x_train_mean)



def read_orig(gap):
    with h5py.File("orig.h5", "r") as hf:
        return hf['image'][::gap], hf['label'][::gap], hf['mean'][:] # [:] to get value from dataset


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print('Provide the path argument for model or model directory')
    # model_dir = sys.argv[1]

    parser = argparse.ArgumentParser("")
    parser.add_argument('-m', '--model', nargs='*',
                        help="specify all models or model directories that is to be attacked")
    parser.add_argument('-s', '--save_dir', help="specify the save directory for attack file", default=None)
    parser.add_argument('-g', '--gap', help='select images with gap ([::10])', type=int, default=1)
    args = parser.parse_args()

    model_locs = args.model
    save_dir = None
    if args.save_dir is None:
        if isinstance(model_locs, list):
            save_dir = model_locs[0]
        else:
            save_dir = model_locs
    else:
        save_dir = args.save_dir
    gap = args.gap
    generate_orig()
    orig_image, orig_label, mean_of_image = read_orig(gap)
    image_shape = mean_of_image.shape
    model, model_type = resnet_ensemble(model_locs, save_dir, image_shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    scores = model.evaluate(orig_image, orig_label, verbose=1)

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