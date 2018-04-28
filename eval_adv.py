import os
import foolbox
import h5py
import numpy as np
import matplotlib.pyplot as plt


def read_orig_selected():
    with  h5py.File('orig_selected.h5', 'r') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][:], hf['pred'][:], hf['label'][:]


def read_adv(name):
    with h5py.File("adv_" + name + ".h5", 'r') as hf:
        return hf['adv'][:], hf['adv_label'][:]


def eval_adv(name, image, pred, label):
    valid = 0
    attack = 0
    adv, adv_label = read_adv(name)
    total = 0
    for i in range(adv_label.shape[0]):
        total += 1
        if np.argmax(pred[i]) == label[i]:
            valid += 1
            if np.argmax(pred[i]) != np.argmax(adv_label[i]):
                attack += 1
    min_val = np.amin(adv - image) * 255
    max_val = np.amax(adv - image) * 255
    avg_val = np.sum(np.abs(adv - image)) / pred.shape[0] / 48 / 48 / 3 * 255
    print("Total for %s attack: %d, Valid Sample: %d, Success: %d, rate: %6.4f" % (
        name, total, valid, attack, attack / valid))
    print("Max value change: %10.8f, Min value change %10.8f, Avg value per pixel per channel: %10.8f" % (
        max_val, min_val, avg_val))


def read_labeled_data(x_test, pred, y_test):
    x = []
    p = []
    y = []
    for i in range(y_test.shape[0]):
        label = y_test[i]
        if label in [0, 14, 33, 34]:
            x.append(x_test[i])
            p.append(pred[i])
            y.append(label)
    return np.array(x), np.array(p), np.array(y)


def visualize_single_graph(name, img, adv, pred, adv_label, label):
    plt.figure()
    image = np.reshape(img, (48, 48, 3))
    adv = np.reshape(adv, (48, 48, 3))
    plt.subplot(3, 3, 1)
    plt.title('Original - pred' + str(pred) + " label: " + str(label))
    plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title(name + ' Adversarial - adv_label: ' + str(adv_label))
    plt.imshow(adv)  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.title('Difference')
    difference = adv - image
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()


def save_to_image(name, img, adv, pred, adv_label, label):
    img = np.reshape(img, (img.shape[0], 32, 32, 3))
    adv = np.reshape(adv, (img.shape[0], 32, 32, 3))
    for i in range(img.shape[0]):
        plt.imsave(str(i) + ".png", img[i], format='png')
        plt.imsave(str(i) + name + ".png", adv[i], format='png')


def example(name, image, pred, label):
    name += "_cifar10_ResNetSVM20v3_model.158.10.0.001_gap"
    adv, adv_label = read_adv(name)
    eval_adv(name, image, pred, label)

    # save_to_image(name, image, adv, pred, adv_label, label)


#    for i in range(10):
#        visualize_single_graph(name, image[i], adv[i], pred[i], adv_label[i], label[i])

def generate_orig_selected(x, pred, y):
    with h5py.File("orig_selected.h5", "w") as hf:
        hf.create_dataset('orig', data=x)
        hf.create_dataset('pred', data=pred)
        hf.create_dataset('label', data=y)


def read_orig(name):
    with  h5py.File("orig_" + name + ".h5", 'r') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][:], hf['pred'][:], hf['label'][:]


if __name__ == '__main__':
    # x_test, pred, y_test = read_orig()
    # x_test, pred, y_test = read_labeled_data(x_test, pred, y_test)
    # generate_orig_selected(x_test, pred, y_test)
    img, pred, label = read_orig('cifar10_ResNetSVM20v3_model.171.5.0.001')

    # choice = np.arange(img.shape[0], step=10)
    # img = np.take(img, choice)
    # pred = np.take(pred, choice)
    # label = np.take(label, choice)

    image = img[::10]
    pred = pred[::10]
    label = label[::10]

    example('DeepFool_L_0', image, pred, label)
    example('DeepFool_L_2', image, pred, label)
    example('LBGFS', image, pred, label)
    example('Iter_Grad', image, pred, label)
    example('Iter_GradSign', image, pred, label)
    example('Local_search', image, pred, label)
    example('Single_Pixel', image, pred, label)
    example('DeepFool_L_INF', image, pred, label)
    example('Gaussian_Blur', image, pred, label)

    # eval_adv('DeepFool_L2', pred, label)
    # eval_adv('DeepFool_L2_INF', pred, label)
    # eval_adv("Single_Pixel", pred, label)
# model = keras.models.load_model("model.h5")
#
# x = np.reshape(x, (1, 3, 48, 48))
# y = np.reshape(y, (1, 3, 48, 48))
# print(model.predict_classes(x))
# print(model.predict_classes(y))
#
# x = np.reshape(x, (48, 48, 3))
# y = np.reshape(y, (48, 48, 3))
