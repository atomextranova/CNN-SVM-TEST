import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import keras
import xlwt
import sys

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
x_train_mean = np.mean(x_train, axis=0)

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


# def eval_adv(model, name, mean, image, pred, label):
def eval_adv(model, image, adv_img, pred_orig, label, model_name, adv_name):
    attack = 0
    adv_label = model.predict(adv_img-mean)
    total = 1000
    for i in range(adv_label.shape[0]):
        if label[i*10] != np.argmax(adv_label[i]):
            attack += 1
    min_val = np.amin(np.abs(adv_img - image)) * 255
    max_val = np.amax(np.abs(adv_img - image)) * 255
    avg_val = np.sum(np.abs(adv_img - image)) / adv_img.size * 255
    print("Total for %s attack: %d, Success: %d, rate: %6.4f" % (
        adv_name, total, attack, attack / total))
    print("Max value change: %10.8f, Min value change %10.8f, Avg value per pixel per channel: %10.8f\n" % (
        max_val, min_val, avg_val))
    return attack, total


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

# def save_to_image(name, img, adv, pred, adv_label, label):
#     img = np.reshape(img, (img.shape[0], 32, 32, 3))
#     adv = np.reshape(adv, (img.shape[0], 32, 32, 3))
#     for i in range(img.shape[0]):
#         plt.imsave(str(i) + ".png", img[i], format='png')
#         plt.imsave(str(i) + name + ".png", adv[i], format='png')


def read_orig():
    with  h5py.File('attack/orig.h5') as hf:
        # value = list(hf.values())
        # print(value)
        return hf['orig'][:], hf['pred'][:], hf['label'][:]


def read_adv_img(model, adv):
    if model == "":
        with h5py.File(file_dir+"/adv_" + adv + "_" + "gap.h5", 'r') as hf:
            return hf['adv'][:]
    else:
        with h5py.File(file_dir+"/adv_" + adv + "_" + model.split("\\")[1] + "_gap.h5", 'r') as hf:
            return hf['adv'][:]

def condition(worksheet_name):
    if len(worksheet_name) >= 30:
        return worksheet_name[-29:]
    else:
        return worksheet_name


if __name__ == '__main__':
    txt_record = open("evaluation_results.txt", 'w')
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

    # label = label[::10]
    with h5py.File("attack/mean.h5", "r") as hf:
        mean = hf['mean'][:]
    # model_list = ["cifar10_ResNet20v1_model.194",
    #               "cifar10_ResNetSVM20v3_model.028.25.0.0001",
    #               "cifar10_ResNetSVM20v3_model.156.35.0.001",
    #               "cifar10_ResNetSVM20v3_model.158.5.0.001",
    #               "cifar10_ResNetSVM20v3_model.158.40.0.001",
    #               "cifar10_ResNetSVM20v3_model.158.10.0.001",
    #               "cifar10_ResNetSVM20v3_model.195.0.5.0.001",
    #               "cifar10_ResNetSVM20v3_model.170.0.1.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.147.0.15.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.121.0.2.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.121.0.25.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.125.0.3.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.195.0.1.0.5.L1.0.001",
    #               # "cifar10_ResNetSVM20v3_model.147.0.1.5.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.122.0.1.10.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.143.0.15.0.5.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.116.0.15.2.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.192.0.15.5.L1.0.001",
    #               "cifar10_ResNetSVM20v3_model.182.0.15.10.L1.0.001",]

    # for root, _, file in os.walk(sys.argv[1]):
    #     if (file.startwith("cifar")):
    #

    file_dir = sys.argv[1]
    file_name = [os.path.splitext(file)[0] for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                  and file.startswith('cifar')
                  and file.endswith('.h5')]

    # xlwt requires less than 31
    # worksheet_name = [name for name in worksheet_name if len(name) < 30]


    model_list = [os.path.join(file_dir, file) for file in file_name]
    model_list_adv = ['clip_ensemble\\ensemble']
    worksheet_name = list(map(condition, file_name))

    # for i, item in enumerate(model_list):
    #     model_list[i] = "attack/"+item

    # # "CNN-SVM-L1-0.1-L2-2","CNN-SVM-L1-0.1-L2-5"
    # worksheet_name = ["CNN", "CNN-SVM-L2-25", "CNN-SVM-L2-35", "CNN-SVM-L2-5", "CNN-SVM-L2-40", "CNN-SVM-L2-10", "CNN-SVM-L2-0.5",
    #                   "CNN-SVM-L1-0.1", "CNN-SVM-L1-0.15", "CNN-SVM-L1-0.2", "CNN-SVM-L1-0.25", "CNN-SVM-L1-0.3",
    #                   "CNN-SVM-L1-0.1-L2-0.5", "CNN-SVM-L1-0.1-L2-10",
    #                   "CNN-SVM-L1-0.15-L2-0.5", "CNN-SVM-L1-0.15-L2-2", "CNN-SVM-L1-0.15-L2-5", "CNN-SVM-L1-0.15-L2-10"]

    # adv_list = ['DeepFool_L_2', 'LBGFS', 'Iter_Grad', 'Iter_GradSign',
    #             'Local_search', 'Single_Pixel', 'DeepFool_L_INF', 'Gaussian_Blur']

    adv_list = ['DeepFool_L_2',
            'DeepFool_L_INF', 'Gaussian_Blur']

    # adv_list = ['DeepFool_L_2',
    #         'DeepFool_L_INF']

    # for model_name in model_list:
    #     for adv_dataset in adv_list:
    file = xlwt.Workbook(encoding = "utf-8")
    # file_real_number = xlwt.Workbook(encoding = "utf-8")
    accuracy = file.add_sheet("Accuracy base line")
    accuracy.write(0, 1, "Loss")
    accuracy.write(0, 2, "Accuracy")
    adv_result_dict = {key: [] for key in adv_list}
    adv_result_cross_dict = {key: [] for key in adv_list}
    for i, model_name in enumerate(model_list):
        model = keras.models.load_model(model_name + ".h5")
        pred = model.predict(x_test[::10])
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
        for j, name in enumerate(model_list_adv):
            print("Using image from model: %s\n" % name)
            # if name == "attack/cifar10_ResNet20v1_model.194":
            if 'cifar10_ResNet20v1_model.194' in name:
                name = ""
            efficiency = []
            for adv_method in adv_list:
                adv_img = read_adv_img(name, adv_method)
                among_adv, among_all = eval_adv(model, img, adv_img, pred, label, name, adv_method)
                efficiency.append(among_adv/among_all)
                if model_name == name:
                    adv_result_dict[adv_method].append(among_adv/among_all)
                else:
                    adv_result_cross_dict[adv_method].append(among_adv/among_all)
            table.write(j+1, 0, name)
            for k, rate in enumerate(efficiency):
                table.write(j+1, k+1, rate)


    txt_record.write("Cross results\n")
    for key, rate in adv_result_cross_dict.items():
        attack_rate = np.array(rate)
        avg = np.average(attack_rate)
        std = np.std(attack_rate)
        report = "{}: average accuracy: {} with variance: {}\n".format(key, avg, std)
        print(report)
        txt_record.write(report)

    txt_record.write("Target results\n")
    for key, rate in adv_result_dict.items():
        attack_rate = np.array(rate)
        avg = np.average(attack_rate)
        std = np.std(attack_rate)
        report = "{}: average accuracy: {} with variance: {}\n".format(key, avg, std)
        print(report)
        txt_record.write(report)
    file.save("report-final-6-10-4.xls")
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
