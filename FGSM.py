import keras
import h5py
import sys
import os
from keras import backend as K
import numpy as np

class Adversarial:

    def __init__(self,  model):
        label_input = K.placeholder(shape=(1,))
        image_input = model.input
        output = model.output
        loss = K.sparse_categorical_crossentropy(label_input, output, from_logits=True)
        grads = K.gradients(loss, image_input)
        grad = grads[0]
        self.grad_func = K.function([image_input, label_input], [grad])

    def iterative_fast_gradient_base(self, image, label, model, preprocessing=(0, 0, 0), clipped=True, sign=False):
        epsilons = np.linspace(0, 0.1, num=51)[1:]



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

    image, _, label = read_orig(10)
    image = np.expand_dims(image, axis=0)

    factor = 0.1
    step_size = 50
    unit_vec = np.ones((32, 32, 3)) / np.sqrt(np.sum(np.square(np.ones((32, 32, 3)))))

    file_dir = sys.argv[1]
    cnn_list = read_model(file_dir, 'cifar')

    # ens_list = read_model(file_dir, 'ens')

    # svm_list = read_model(file_dir, 'cifar', True)

    for model in cnn_list:
        fast


            # cur_grad_min = cur_grad.min(axis=(0, 1), keepdims=True)
            # cur_grad_max = cur_grad.max(axis=(0, 1), keepdims=True)
            # cur_grad_norm = (cur_grad - cur_grad_min) / (cur_grad_max - cur_grad_min)