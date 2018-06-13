import keras
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import h5py
import time
import foolbox
import os
import sys
import threading

subtract_pixel_mean = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

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

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)


def generate_orig(name, x, pred, y):
    with h5py.File("orig" + name + ".h5", "w") as hf:
        hf.create_dataset('orig', data=x)
        hf.create_dataset('pred', data=pred)
        hf.create_dataset('label', data=y)


def attack_wrapper(model, model_name, attack, name, gap, lock, part=False):
    # root_dir = os.getcwd()
    # save_base_path = os.path.join(root_dir, "adv_image/{}".format(model_base_name))
    save_base_path = "{}/{}".format(sys.argv[1], model_name)

    # Race conditon here: Let pass will not work
    # 1) other error other than
    #   a) if except OSError as e:
    #      if e.errno != errno.EEXIST:
    #          raise
    #      will not work either cuz of 2)
    # 2) created and removed during the same check, raise error but nothing there
    have_it = lock.acquire()
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    lock.release()
    name += "_" + model_name
    adv = []
    # for i in range(x_test.shape[0]):
    # for i in range(int(x_test.shape[0] / 40)):
    record = open(os.path.join(save_base_path, "{}.txt").format(name), 'w')
    record.write("--- " + name + " started ---\n")
    start = time.time()
    count = 0
    for i in range(int(x_test.shape[0] / gap)):
        # for i in range(1):
        # for i in range(10):
        if i % 10 == 0:
            record.write("Generateing %d images with gap %d\n" % (i, gap))
        adv_image = attack(x_test[i * gap], y_test[i * gap])
        # if adv_image == None:
        #     adv_image
        if adv_image is not None:
            adv.append(adv_image)
        else:
            record.write("Fail to generate adv image. Appending original image.\n")
            adv.append(x_test[i * gap])
            count += 1
    adv = np.array(adv, 'float32')
    record.write("--- " + name + " %s seconds ---\n" % (time.time() - start))
    record.write("Sucessfully generated %d images with gap %d, including %d original images\n" % (
        int(adv.shape[0]), gap, count))
    if part:
        name += "_part"
    elif gap != 1:
        name += "_gap"
    with h5py.File(os.path.join(save_base_path, "adv_" + name + ".h5"), "w") as hf:
        hf.create_dataset('adv', data=(adv + x_train_mean))
        hf.create_dataset('adv_label', data=model.predict(adv))
    record.close()
    return


gap = 10


def attack_group_1(model_adv, model, model_name, lock):
    # start = time.time()
    attack_deep_fool_l2 = foolbox.attacks.DeepFoolL2Attack(model_adv)

    attack_DFL_INF = foolbox.attacks.DeepFoolLinfinityAttack(model_adv)

    attack_wrapper(model, model_name, attack_deep_fool_l2, "DeepFool_L_2", gap, lock)
    attack_wrapper(model, model_name, attack_DFL_INF, 'DeepFool_L_INF', gap, lock)


    # attack_LBFGSAttack = foolbox.attacks.LBFGSAttack(model_adv)
    # attack_wrapper(model, model_name, attack_LBFGSAttack, 'LBGFS', gap, lock)
    #
    attack_GaussianBlur = foolbox.attacks.GaussianBlurAttack(model_adv)
    attack_wrapper(model, model_name, attack_GaussianBlur, "Gaussian_Blur", gap, lock)
    # # print("--- " + str(1) + "takes %s seconds ---\n" % (time.time() - start))


def attack_group_2(model_adv, model, model_name, lock):
    # start = time.time()
    attack_IterGradSign = foolbox.attacks.IterativeGradientSignAttack(model_adv)
    attack_wrapper(model, model_name, attack_IterGradSign, "Iter_GradSign", gap, lock)
    # print("--- " + str(2) + "takes %s seconds ---\n" % (time.time() - start))


def attack_group_3(model_adv, model, model_name, lock):
    # start = time.time()
    attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
    attack_wrapper(model, model_name, attack_IterGrad, "Iter_Grad", gap, lock)
    # print("--- " + str(3) + "takes %s seconds ---\n" % (time.time() - start))


def attack_group_4(model_adv, model, model_name, lock):
    # start = time.time()
    attack_Local = foolbox.attacks.LocalSearchAttack(model_adv)
    attack_Single_Pixel = foolbox.attacks.SinglePixelAttack(model_adv)
    attack_wrapper(model, model_name, attack_Local, "Local_Search", gap, lock)
    attack_wrapper(model, model_name, attack_Single_Pixel, "Single_Pixel", gap, lock)
    # print("--- " + str(4) + "takes %s seconds ---\n" % (time.time() - start))


# def generate_adv(model, model_name):
#     # try:
#     #     with h5py.File("orig.h5", "r") as hf:
#     #
#     # generate_orig("_" , x_test + x_train_mean, pred, y_test)
#     model_adv = foolbox.models.KerasModel(model, bounds=(-1, 1), preprocessing=((0, 0, 0), 1))
#
#     attack_deep_fool_l2 = foolbox.attacks.DeepFoolL2Attack(model_adv)
#
#     attack_DFL_INF = foolbox.attacks.DeepFoolLinfinityAttack(model_adv)
#     attack_DFL_0 = foolbox.attacks.DeepFoolAttack(model_adv)
#     attack_IterGradSign = foolbox.attacks.IterativeGradientSignAttack(model_adv)
#
#     attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
#     attack_LBFGSAttack = foolbox.attacks.LBFGSAttack(model_adv)
#
#     attack_Local = foolbox.attacks.LocalSearchAttack(model_adv)
#
#     attack_GaussianBlur = foolbox.attacks.GaussianBlurAttack(model_adv)
#
#     attack_Single_Pixel = foolbox.attacks.SinglePixelAttack(model_adv)
#
#     attack_wrapper(model, model_name, attack_deep_fool_l2, "DeepFool_L_2", gap)
#     attack_wrapper(model, model_name, attack_DFL_INF, 'DeepFool_L_INF', gap)
#     attack_wrapper(model, model_name, attack_DFL_0, "DeepFool_L_0", gap)
#
#     attack_wrapper(model, model_name, attack_LBFGSAttack, 'LBGFS', gap)
#     attack_wrapper(model, model_name, attack_IterGrad, "Iter_Grad", gap)
#     attack_wrapper(model, model_name, attack_IterGradSign, "Iter_GradSign", gap)
#
#     attack_wrapper(model, model_name, attack_Local, "Local_Search", gap)
#     attack_wrapper(model, model_name, attack_Single_Pixel, "Single_Pixel", gap)
#
#     attack_wrapper(model, model_name, attack_GaussianBlur, "Gaussian_Blur", gap)


# attack_BoundaryAttack = foolbox.attacks.BoundaryAttack(model_adv)
# attack_wrapper(attack_BoundaryAttack, "Boundary", gap)

def attack(model_dir, model_name):
    print(model_name, " started")
    model_name = os.path.splitext(model_name)[0]
    start = time.time()
    model = keras.models.load_model(model_dir)
    # make thread ready manually
    model._make_predict_function()
    model_adv = foolbox.models.KerasModel(model, bounds=(-1, 1), preprocessing=((0,0,0), 1))

    thread_list = []
    my_args_dict = dict(model_adv=model_adv, model=model, model_name=model_name, lock=threading.Lock())
    thread_list.append(threading.Thread(target=attack_group_1, kwargs=my_args_dict))
    # thread_list.append(threading.Thread(target=attack_group_2, kwargs=my_args_dict))
    # thread_list.append(threading.Thread(target=attack_group_3, kwargs=my_args_dict))
    # thread_list.append(threading.Thread(target=attack_group_4, kwargs=my_args_dict))

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    print("--- " + model_name + "takes %s seconds ---\n" % (time.time() - start))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Provide the path argument for model or model directory')
    model_dir = sys.argv[1]

    if os.path.isfile(model_dir):
        model_name = os.path.basename(model_dir)
        attack(model_dir, model_name)
    elif os.path.isdir(model_dir):
        for root, _, files in os.walk(model_dir):
            for model_name in files:
                model_dir = os.path.join(root, model_name)
                attack(model_dir, model_name)


