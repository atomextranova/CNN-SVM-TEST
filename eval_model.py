import keras
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import h5py
import time
import foolbox

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

model = keras.models.load_model("cifar10_ResNetSVM20v3_model.171.5.0.001.h5")
scores = model.evaluate(x_test, keras.utils.to_categorical(y_test, 10), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.summary()

pred = model.predict(x_test)


def generate_orig(name, x, pred, y):
    with h5py.File("orig" + name + ".h5", "w") as hf:
        hf.create_dataset('orig', data=x)
        hf.create_dataset('pred', data=pred)
        hf.create_dataset('label', data=y)


def attack_wrapper(attack, name, gap=1, part=False):
    name += "_cifar10_ResNetSVM20v3_model.158.10.0.001"
    adv = []
    # for i in range(x_test.shape[0]):
    # for i in range(int(x_test.shape[0] / 40)):
    print("--- " + name + " started ---")
    start = time.time()
    count = 0
    for i in range(int(x_test.shape[0] / gap)):
        # for i in range(1):
        # for i in range(10):
        if i % 10 == 0:
            print("Generateing %d images with gap %d" % (i, gap))
        adv_image = attack(x_test[i * gap], y_test[i * gap])
        # if adv_image == None:
        #     adv_image
        if adv_image is not None:
            adv.append(adv_image)
        else:
            print("Invalid adv image. Appending original image.")
            adv.append(x_test[i * gap])
            count += 1
    adv = np.array(adv, 'float32')
    print("--- " + name + " %s seconds ---" % (time.time() - start))
    print("Sucessfully generated %d images with gap %d, including %d original images" % (
    int(adv.shape[0]), gap, count))
    if part:
        name += "_part"
    elif gap != 1:
        name += "_gap"
    with h5py.File("adv_" + name + ".h5", "w") as hf:
        hf.create_dataset('adv', data=(adv + x_train_mean))
        hf.create_dataset('adv_label', data=model.predict(adv))


# try:
#     with h5py.File("orig.h5", "r") as hf:
#
generate_orig("_cifar10_ResNetSVM20v3_model.171.5.0.001", x_test + x_train_mean, pred, y_test)
model_adv = foolbox.models.KerasModel(model, bounds=(-1, 1), preprocessing=((0, 0, 0), 1))

attack_deep_fool_l2 = foolbox.attacks.DeepFoolL2Attack(model_adv)

attack_DFL_INF = foolbox.attacks.DeepFoolLinfinityAttack(model_adv)
attack_DFL_0 = foolbox.attacks.DeepFoolAttack(model_adv)
attack_IterGradSign = foolbox.attacks.IterativeGradientSignAttack(model_adv)

attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
attack_LBFGSAttack = foolbox.attacks.LBFGSAttack(model_adv)

attack_Local = foolbox.attacks.LocalSearchAttack(model_adv)

attack_GaussianBlur = foolbox.attacks.GaussianBlurAttack(model_adv)

attack_Single_Pixel = foolbox.attacks.SinglePixelAttack(model_adv)

attack_wrapper(attack_deep_fool_l2, "DeepFool_L_2", 10)
attack_wrapper(attack_DFL_INF, 'DeepFool_L_INF', 10)
attack_wrapper(attack_DFL_0, "DeepFool_L_0", 10)

attack_wrapper(attack_LBFGSAttack, 'LBGFS', 10)
attack_wrapper(attack_IterGrad, "Iter_Grad", 10)
attack_wrapper(attack_IterGradSign, "Iter_GradSign", 10)

attack_wrapper(attack_Local, "Local_Search", 10)
attack_wrapper(attack_Single_Pixel, "Single_Pixel", 10)

attack_wrapper(attack_GaussianBlur, "Gaussian_Blur", 10)
# attack_BoundaryAttack = foolbox.attacks.BoundaryAttack(model_adv)
# attack_wrapper(attack_BoundaryAttack, "Boundary", 10)
