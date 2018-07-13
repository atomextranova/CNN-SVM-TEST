import keras
import h5py
import sys
import os
from keras import backend as K
import numpy as np


def read_model(file_dir, key, svm=False):

    if not svm:
        model_dir = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                      and file.startswith(key) and 'SVM' not in file
                      and file.endswith('.h5')]
        model_list = [keras.models.load_model(os.path.join(file_dir, file)) for file in model_dir]
        for model, model_name in zip(model_list, model_dir):
            model.name = model_name

    else:
        model_dir = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                      and file.startswith(key) and 'SVM' in file
                      and file.endswith('.h5')]
        model_list = [keras.models.load_model(os.path.join(file_dir, file)) for file in model_dir]
        for model, model_name in zip(model_list, model_dir):
            model.name = model_name
    return model_list

def read_orig(gap=1):
    with  h5py.File('attack/orig.h5') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][::gap], hf['pred'][::gap], hf['label'][::gap]

if __name__ == '__main__':

    with h5py.File("attack/mean.h5", "r") as hf:
        mean = hf['mean'][:]

    image, _, label = read_orig(10000)
    image = np.expand_dims(image, axis=0)

    factor = 0.1
    step_size = 50
    unit_vec = np.ones((32, 32, 3)) / np.sqrt(np.sum(np.square(np.ones((32, 32, 3)))))

    file_dir = sys.argv[1]
    cnn_list = read_model(file_dir, 'cifar')

    # ens_list = read_model(file_dir, 'ens')

    # svm_list = read_model(file_dir, 'cifar', True)



    for model in cnn_list:

        label_input = K.placeholder(shape=(1,))
        image_input = model.input
        output = model.output
        loss = K.sparse_categorical_crossentropy(label_input, output, from_logits=True)
        grads = K.gradients(loss, image_input)
        grad = grads[0]
        grad_func = K.function([image_input, label_input], [grad])

        for img, lab in zip(image, label):
            image_list = []
            cur_grad = grad_func([img, lab])
            cur_grad_normalized = cur_grad / np.sqrt(np.sum(np.square(cur_grad)))
            for step_grad in range(step_size):
                for step_unit in range(step_size):
                    temp_img = img + np.multiply(step_grad,cur_grad) + np.multiply(step_unit,unit_vec)
                    temp_img_clipped = np.clip(temp_img, 0, 1)
                    image_list.extend(temp_img_clipped)
                    # temp_img = np.expand_dims(img, axis=0) + step_grad *
            pass
            image_array = np.concatenate(image_list)
            result_list = model.predict(image_array)
            print(result_list)
            pass
            pass
            # cur_grad_min = cur_grad.min(axis=(0, 1), keepdims=True)
            # cur_grad_max = cur_grad.max(axis=(0, 1), keepdims=True)
            # cur_grad_norm = (cur_grad - cur_grad_min) / (cur_grad_max - cur_grad_min)





