import keras
import h5py
import sys
import os
from keras import backend as K
import numpy as np
from collections import Iterable
import logging
import time


class Adversarial:

    def __init__(self, cur_model, softmax=False, convert_activation=False, label_input=None):

        if convert_activation:
            cur_model = Adversarial.activation_to_linear(cur_model)
        self.model = cur_model
        self.grad_func = Adversarial.make_gradient(cur_model, softmax, convert_activation, label_input)

    @staticmethod
    def activation_to_linear(model):
        model.layers[-1].activation = keras.layers.activations.linear
        return model

    @staticmethod
    def make_gradient(model, softmax=False, convert_activation=False, label_input=None):
        if label_input is None:
            label_input = K.placeholder(shape=(1,))
        image_input = model.get_input_at(0)
        output = model.get_output_at(0)
        if softmax and not convert_activation:
            logging.warning(
                'Rely on instable number conversion. Remove softmax activation to ensure numeric stability')
            loss = K.sparse_categorical_crossentropy(label_input, output, from_logits=False)
        else:
            loss = K.sparse_categorical_crossentropy(label_input, output, from_logits=True)
        grads = K.gradients(loss, image_input)
        grad = grads[0]
        return K.function([image_input, label_input], [grad])

    @staticmethod
    def get_list_of_component(model):
        return model.layers[1:-1]

    @staticmethod
    def root_mean_square_deviation(orig_image, adv_image):
        return np.sqrt(np.sum(np.square(orig_image - adv_image) / np.array(orig_image.size)))

    # @staticmethod
    # def l2_distance():

    def gradient(self, image, label):
        return self.grad_func([image, label])[0]

    def predict(self, image):
        return np.argmax(self.model.predict(image), axis=1)

    def raw_predict(self, image):
        return self.model.predict(image)

    def deepfool_base(self, image, label):
        pass



    def _iterative_gradient_base(self, image, label, epsilons=None, epsilon_number=51, epsilon_space=(0, 0.1),
                                 steps=100, preprocessing=(0, 0, 0), clip=True, val_range=(0, 1), sign=False,
                                 optimize=False):
        image = np.expand_dims(image, axis=0) - preprocessing
        if self.predict(image) != label:
            logging.warning('Already wrong. Not adding any noise')
            return None
        else:
            min_val, max_val = val_range
            min_val_eps, max_val_eps = epsilon_space
            if not isinstance(epsilons, Iterable):
                epsilons = np.linspace(min_val_eps, max_val_eps, num=epsilon_number)[1:]

            for epsilon in epsilons:
                perturbed_image = image
                for _ in range(steps):
                    cur_grad = self.gradient(perturbed_image, label)
                    if sign:
                        cur_grad = np.sign(cur_grad)
                    perturbed_image += cur_grad * epsilon
                    if clip:
                        perturbed_image = np.clip(perturbed_image + preprocessing, min_val, max_val) - preprocessing
                    if label != self.predict(perturbed_image):
                        return np.squeeze(perturbed_image + preprocessing, axis=0)
            logging.warning('Does not found adversarial given current settings')
            return None

    def iterative_gradient_sign_method(self, image, label, epsilons=None, epsilon_number=51, epsilon_space=(0, 0.1),
                                       steps=100, preprocessing=(0, 0, 0), clip=True, val_range=(0, 1), optimize=False):
        return self._iterative_gradient_base(image, label, epsilons=epsilons, epsilon_number=epsilon_number,
                                             epsilon_space=epsilon_space,
                                             steps=steps, preprocessing=preprocessing, clip=clip, sign=True,
                                             val_range=val_range,
                                             optimize=optimize)

    def iterative_gradient_method(self, image, label, epsilons=None, epsilon_number=51, epsilon_space=(0, 0.1),
                                  steps=100, preprocessing=(0, 0, 0), clip=True, val_range=(0, 1), optimize=False):
        return self._iterative_gradient_base(image, label, epsilons=epsilons, epsilon_number=epsilon_number,
                                             epsilon_space=epsilon_space,
                                             steps=steps, preprocessing=preprocessing, clip=clip, sign=False,
                                             val_range=val_range,
                                             optimize=optimize)

    # def iterative_gradient_sign_method(self, image, label, epsilons=None, epsilon_number=51, epsilon_space=(0, 0.1),
    #                                    steps=100, preprocessing=(0, 0, 0), clip=True, range=(0, 1), optimize=False):
    #     image -= preprocessing
    #     min_val, max_val = range
    #     min_val_eps, max_val_eps = epsilon_space
    #     if not isinstance(epsilons, Iterable):
    #         epsilons = np.linspace(min_val_eps, max_val_eps, num=epsilon_number)[1:]
    #
    #     for epsilon in epsilons:
    #         perturbed_image = np.expand_dims(image, axis=0)
    #         for _ in range(steps):
    #             cur_grad = self.gradient(perturbed_image, label)
    #             cur_grad = np.sign(cur_grad)
    #             perturbed_image += cur_grad * epsilon
    #             if clip:
    #                 perturbed_image = np.clip(perturbed_image + preprocessing, min_val, max_val) - preprocessing
    #             if label != self.predict(perturbed_image):
    #                 return perturbed_image + preprocessing
    #     logging.warning('Does not found adversarial given current settings')
    #     return None

    def iterative_gradient_method_ensemble(self, image, label, epsilons=None, epsilon_number=101,
                                           epsilon_space=(0, 0.1),
                                           steps=1000, preprocessing=(0, 0, 0), clip=True, val_range=(0, 1), sign=False,
                                           optimize=False):
        image = np.expand_dims(image, axis=0) - preprocessing
        if self.predict(image) != label:
            logging.warning('Already wrong. Not adding any noise. Return None')
            return None
        else:
            min_val, max_val = val_range
            min_val_eps, max_val_eps = epsilon_space
            if not isinstance(epsilons, Iterable):
                epsilons = np.linspace(min_val_eps, max_val_eps, num=epsilon_number)[1:]

            model_list = Adversarial.get_list_of_component(self.model)
            grad_list = []
            for child_model in model_list:
                grad_list.append(Adversarial.make_gradient(child_model))

            for epsilon in epsilons:
                perturbed_image = image
                for _ in range(steps):
                    cur_grad = 0
                    grad_num = 0
                    for cur_gradient_func, cur_model in zip(grad_list, model_list):
                        cur_pred = np.argmax(cur_model.predict(perturbed_image), axis=1)
                        if cur_pred == label:
                            cur_grad += cur_gradient_func([perturbed_image, label])[0]
                        else:
                            break
                    if sign:
                        cur_grad = np.sign(cur_grad)
                    perturbed_image += cur_grad * epsilon
                    if clip:
                        perturbed_image = np.clip(perturbed_image + preprocessing, min_val, max_val) - preprocessing
                    if self.predict(perturbed_image) != label:
                        return np.squeeze(perturbed_image + preprocessing, axis=0)
            logging.warning('Does not found adversarial given current settings. Return None.')
            return None

    # def iterative_gradient_method_ensemble(self, image, label, epsilons=None, epsilon_number=101,
    #                                        epsilon_space=(0, 0.1),
    #                                        steps=100, preprocessing=(0, 0, 0), clip=True, val_range=(0, 1), sign=False,
    #                                        optimize=False):
    #     image = np.expand_dims(image, axis=0) - preprocessing
    #     if self.predict(image) != label:
    #         logging.warning('Already wrong. Not adding any noise. Return None')
    #         return None
    #     else:
    #         min_val, max_val = val_range
    #         min_val_eps, max_val_eps = epsilon_space
    #         if not isinstance(epsilons, Iterable):
    #             epsilons = np.linspace(min_val_eps, max_val_eps, num=epsilon_number)[1:]
    #
    #         model_list = Adversarial.get_list_of_component(self.model)
    #         grad_list = []
    #         for child_model in model_list:
    #             grad_list.append(Adversarial.make_gradient(child_model))
    #
    #         for epsilon in epsilons:
    #             perturbed_image = image
    #             for _ in range(steps):
    #                 cur_grad = 0
    #                 grad_num = 0
    #                 for cur_gradient_func, cur_model in zip(grad_list, model_list):
    #                     cur_pred = np.argmax(cur_model.predict(perturbed_image), axis=1)
    #                     if cur_pred == label:
    #                         cur_grad += cur_gradient_func([perturbed_image, label])[0]
    #                         grad_num += 1
    #                 print(grad_num)
    #                 if sign:
    #                     cur_grad = np.sign(cur_grad)
    #                 perturbed_image += cur_grad * epsilon
    #                 if clip:
    #                     perturbed_image = np.clip(perturbed_image + preprocessing, min_val, max_val) - preprocessing
    #                 if self.predict(perturbed_image) != label:
    #                     return np.squeeze(perturbed_image + preprocessing, axis=0)
    #         logging.warning('Does not found adversarial given current settings. Return None.')
    #         return None


def read_model(file_dir, key, svm=False):
    if not svm:
        model_dir = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                     and file.startswith(key) and 'SVM' not in file
                     and file.endswith('.h5')]
        model_list = [keras.models.load_model(os.path.join(file_dir, file)) for file in model_dir]
        for model, model_name in zip(model_list, model_dir):
            model.name = os.path.splitext(model_name)[0]

    else:
        model_dir = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                     and file.startswith(key) and 'SVM' in file
                     and file.endswith('.h5')]
        model_list = [keras.models.load_model(os.path.join(file_dir, file)) for file in model_dir]
        for model, model_name in zip(model_list, model_dir):
            model.name = os.path.splitext(model_name)[0]
    return model_list


def read_orig(gap=1):
    with  h5py.File('attack/orig.h5') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][::gap], hf['pred'][::gap], hf['label'][::gap]


def generate_adv_for_list(model_list, file_dir, ens=False):
    for model in model_list:
        file = open(os.path.join(file_dir, 'adv_{}_{}_gap.txt'.format('Iter_Grad', model.name)), 'w')
        print('--- Model {} started ---'.format(model.name))
        start = time.time()
        model_adv = Adversarial(model, convert_activation=True)
        adv_list = []
        adv_list_ensemble = []
        success = 0
        success_ensemble = 0
        for img, lab in zip(image, label):
            adv_image = model_adv.iterative_gradient_method(image=img + mean, label=lab, preprocessing=mean)
            if adv_image is None:
                adv_list.append(img)
                logging.warning('Invalid adversarial graph')
            else:
                adv_list.append(adv_image)
                if model_adv.predict(np.expand_dims(adv_image - mean, axis=0)) != lab:
                    success += 1
            if ens:
                adv_image_ensemble = model_adv.iterative_gradient_method_ensemble(image=img + mean, label=lab,
                                                                                  preprocessing=mean)
                if adv_image_ensemble is None:
                    adv_list_ensemble.append(img)
                    logging.warning('Invalid adversarial graph')
                else:
                    adv_list_ensemble.append(adv_image_ensemble)
                    if model_adv.predict(np.expand_dims(adv_image - mean, axis=0)) != lab:
                        success_ensemble += 1

        file.write('Sum RMSD: {} \n'.format(Adversarial.root_mean_square_deviation(image + mean, np.array(adv_list))))
        file.write(str(np.sum(np.abs(image+mean-np.array(adv_list))) / np.array(adv_list).size))
        if ens:
            file.write('Ensemble RMSD: {} \n'.format(
                Adversarial.root_mean_square_deviation(image + mean, np.array(adv_list_ensemble))))
            file.write(str(np.sum(np.abs(image + mean - np.array(adv_list_ensemble))) / np.array(adv_list_ensemble).size))

        completion_msg = "--- {} takes {} seconds ---\n".format(model.name, (time.time() - start))
        print(completion_msg)
        file.close()

        with h5py.File(os.path.join(file_dir, 'adv_{}_{}_gap.h5'.format('Iter_Grad', model.name)), "w") as hf:
            hf.create_dataset(name='sum', data=np.array(adv_list))
            hf.create_dataset(name='ens', data=np.array(adv_list_ensemble))


if __name__ == '__main__':
    with h5py.File("attack/mean.h5", "r") as hf:
        mean = np.array(hf['mean'][:])

    image, _, label = read_orig(10)

    file_dir = sys.argv[1]
    cnn_list = read_model(file_dir, 'cifar')

    ens_list = read_model(file_dir, 'ens')
    generate_adv_for_list(cnn_list, file_dir)
    generate_adv_for_list(ens_list, file_dir)

    # svm_list = read_model(file_dir, 'cifar', True)
