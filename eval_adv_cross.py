import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import keras
import xlwt


# def eval_adv(model, name, mean, image, pred, label):
def eval_adv(model, image, adv_img, pred_orig, label, model_name, adv_name):
    valid = 0
    attack = 0
    adv_label = model.predict(adv_img - mean)
    total = 0
    for i in range(adv_label.shape[0]):
        total += 1
        if np.argmax(pred_orig[i]) == label[i]:
            valid += 1
            if np.argmax(pred_orig[i]) != np.argmax(adv_label[i]):
                attack += 1
    min_val = np.amin(adv_img - image) * 255
    max_val = np.amax(adv_img - image) * 255
    avg_val = np.sum(np.abs(adv_img - image)) / adv_img.shape[0] / adv_img.shape[1] / adv_img.shape[2] / adv_img.shape[
        3] * 255
    if valid == 0:
        valid = 1
    print("Total for %s attack: %d, Valid Sample: %d, Success: %d, rate: %6.4f" % (
        adv_name, total, valid, attack, attack / valid))
    print("Max value change: %10.8f, Min value change %10.8f, Avg value per pixel per channel: %10.8f\n" % (
        max_val, min_val, avg_val))
    return attack / valid


# def read_labeled_data(x_test, pred, y_test):
#     x = []
#     p = []
#     y = []
#     for i in range(y_test.shape[0]):
#         label = y_test[i]
#         if label in [0, 14, 33, 34]:
#             x.append(x_test[i])
#             p.append(pred[i])
#             y.append(label)
#     return np.array(x), np.array(p), np.array(y)


# def visualize_single_graph(name, img, adv, pred, adv_label, label):
#     plt.figure()
#     image = np.reshape(img, (48, 48, 3))
#     adv = np.reshape(adv, (48, 48, 3))
#     plt.subplot(3, 3, 1)
#     plt.title('Original - pred' + str(pred) + " label: " + str(label))
#     plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
#     plt.axis('off')
#
#     plt.subplot(3, 3, 5)
#     plt.title(name + ' Adversarial - adv_label: ' + str(adv_label))
#     plt.imshow(adv)  # ::-1 to convert BGR to RGB
#     plt.axis('off')
#
#     plt.subplot(3, 3, 9)
#     plt.title('Difference')
#     difference = adv - image
#     plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
#     plt.axis('off')
#
#     plt.show()

def save_to_image(name, img, adv, pred, adv_label, label):
    img = np.reshape(img, (img.shape[0], 32, 32, 3))
    adv = np.reshape(adv, (img.shape[0], 32, 32, 3))
    for i in range(img.shape[0]):
        plt.imsave(str(i) + ".png", img[i], format='png')
        plt.imsave(str(i) + name + ".png", adv[i], format='png')


def generate_orig_selected(x, pred, y):
    with h5py.File("orig_selected.h5", "w") as hf:
        hf.create_dataset('orig', data=x)
        hf.create_dataset('pred', data=pred)
        hf.create_dataset('label', data=y)


def read_orig():
    with  h5py.File('orig.h5') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][:], hf['pred'][:], hf['label'][:]


def read_adv_img(model, adv):
    if model == "":
        with h5py.File("adv_" + adv + "_" + "gap.h5", 'r') as hf:
            return hf['adv'][:]
    else:
        with h5py.File("adv_" + adv + "_" + model + "_gap.h5", 'r') as hf:
            return hf['adv'][:]


if __name__ == '__main__':
    # x_test, pred, y_test = read_orig()
    # x_test, pred, y_test = read_labeled_data(x_test, pred, y_test)
    # generate_orig_selected(x_test, pred, y_test)
    image, _, label = read_orig()
    label_ex = keras.utils.to_categorical(label, 10)

    # choice = np.arange(img.shape[0], step=10)
    # img = np.take(img, choice)
    # pred = np.take(pred, choice)
    # label = np.take(label, choice)

    img = image[::10]

    label = label[::10]
    with h5py.File("mean.h5", "r") as hf:
        mean = hf['mean'][:]
    model_list = ["cifar10_ResNet20v1_model.194",
                  "cifar10_ResNetSVM20v3_model.028.25.0.0001",
                  "cifar10_ResNetSVM20v3_model.156.35.0.001",
                  "cifar10_ResNetSVM20v3_model.158.5.0.001",
                  "cifar10_ResNetSVM20v3_model.158.40.0.001",
                  "cifar10_ResNetSVM20v3_model.158.10.0.001",
                  "cifar10_ResNetSVM20v3_model.195.0.5.0.001"]

    worksheet_name = ["CNN", "CNN-SVM-25", "CNN-SVM-35", "CNN-SVM-5", "CNN-SVM-40", "CNN-SVM-10", "CNN-SVM-0.5"]

    adv_list = ['DeepFool_L_0', 'DeepFool_L_2', 'LBGFS', 'Iter_Grad', 'Iter_GradSign',
                'Local_search', 'Single_Pixel', 'DeepFool_L_INF', 'Gaussian_Blur']

    # for model_name in model_list:
    #     for adv_dataset in adv_list:
    file = xlwt.Workbook(encoding = "utf-8")
    accuracy = file.add_sheet("Accuracy base line")
    accuracy.write(0, 1, "Loss")
    accuracy.write(0, 2, "Accuracy")
    for i, model_name in enumerate(model_list):
        model = keras.models.load_model(model_name + ".h5")
        pred = model.predict(img - mean)
        print("--- Evaluation: %s, started ---\n" % (model_name))
        loss, acc = model.evaluate(image-mean, label_ex, verbose=0)
        print('Test loss:', loss)
        print('Test accuracy:', acc)
        accuracy.write(i + 1, 0, model_name)
        accuracy.write(i + 1, 1, loss)
        accuracy.write(i + 1, 2, acc)
        table = file.add_sheet(worksheet_name[i])
        for l, adv_name in enumerate(adv_list):
            table.write(0, l+1, adv_name)
        for j, name in enumerate(model_list):
            print("Using image from model: %s\n" % name)
            if name == "cifar10_ResNet20v1_model.194":
                name = ""
            efficiency = []
            for adv_method in adv_list:
                adv_img = read_adv_img(name, adv_method)
                efficiency.append(eval_adv(model, img, adv_img, pred, label, name, adv_method))
            table.write(j+1, 0, name)
            for k, rate in enumerate(efficiency):
                table.write(j+1, k+1, rate)

    file.save("report-final.xlsx")
    # example('DeepFool_L_0', image, pred, label)
    # example('DeepFool_L_2', image, pred, label)
    # example('LBGFS', image, pred, label)
    # example('Iter_Grad', image, pred, label)
    # example('Iter_GradSign', image, pred, label)
    # example('Local_search', image, pred, label)
    # example('Single_Pixel', image, pred, label)
    # example('DeepFool_L_INF', image, pred, label)
    # example('Gaussian_Blur', image, pred, label)

    # eval_adv('DeepFool_L2', pred, label)
    # eval_adv('DeepFool_L2_INF', pred, label)
    # eval_adv("Single_Pixel", pred, label)
