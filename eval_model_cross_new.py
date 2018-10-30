import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import keras
import xlwt
import sys
import argparse

# # Load the CIFAR10 data.
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
#
# # Input image dimensions.
# input_shape = x_train.shape[1:]
#
# # Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
#
# # If subtract pixel mean is enabled
# x_train_mean = np.mean(x_train, axis=0)
#
# x_train -= x_train_mean
# x_test -= x_train_mean



def array_to_scalar(arr):
    list = []
    for item in arr:
        list.append(np.asscalar(item))
    return np.array(list)


# # Convert class vectors to binary class matrices.
# y_train = array_to_scalar(y_train)
# y_test = array_to_scalar(y_test)
#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)


# def eval_adv(model, name, mean, image, pred, label):
def eval_adv(model, image, adv_img, pred_orig, label, model_name, adv_name):
    attack = 0
    adv_label = model.predict(adv_img-mean)
    total = 1000
    for i in range(adv_label.shape[0]):
        if label[i] != np.argmax(adv_label[i]):
            attack += 1
    min_val = np.amin(np.abs(adv_img - image)) * 255
    max_val = np.amax(np.abs(adv_img - image)) * 255
    avg_val = np.sum(np.abs(adv_img - image)) / adv_img.size * 255
    avg_var = np.var(np.abs(adv_img - image) * 255)
    print("Total for %s attack: %d, Success: %d, rate: %6.4f" % (
        adv_name, total, attack, attack / total))
    print("Max value change: %10.8f, Min value change %10.8f, Avg value per pixel per channel: %10.8f with variance %10.8f\n" % (
        max_val, min_val, avg_val, avg_var))
    return attack, total

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


def generate_orig():
    if not os.path.exists('orig.h5'):
        subtract_pixel_mean = True

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

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


def read_adv_img(model, adv):
    if model == "":
        with h5py.File(file_dir+"/adv_" + adv + "_" + "gap.h5", 'r') as hf:
            return hf['adv'][:]
    else:
        with h5py.File(file_dir+"/adv_" + adv + "_" + model.split("\\")[1] + "_gap.h5", 'r') as hf:
            return hf['adv'][:]

def process_worksheet_name(worksheet_name):
    if len(worksheet_name) >= 30:
        return worksheet_name[-29:]
    else:
        return worksheet_name


if __name__ == '__main__':

    file_dir = sys.argv[1]
    model_dirs = sys.argv[2]
    gap = sys.argv[3]

    txt_record = open("evaluation_results.txt", 'w')
    generate_orig()
    image, label, mean = read_orig(gap)
    label_ex = keras.utils.to_categorical(label, 10)

    # load model
    model_dict = {}
    for model_file in os.listdir(model_dirs):
        model_dir = os.path.join(model_dirs, model_file)
        model_dict[model_file] = keras.models.load_model(model_dir)

    file_name = [os.path.splitext(file)[0] for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
                  and file.startswith('cifar')
                  and file.endswith('.h5')]
    image_model_list = [os.path.join(file_dir, file) for file in file_name]
    worksheet_name = list(map(process_worksheet_name, file_name))


    # adv_list = ['DeepFool_L_2', 'LBGFS', 'Iter_Grad', 'Iter_GradSign',
    #             'Local_search', 'Single_Pixel', 'DeepFool_L_INF', 'Gaussian_Blur']

    adv_list = ['DeepFool_L_2']
            # ,'DeepFool_L_INF', 'Gaussian_Blur',  'Iter_Grad', 'LBGFS', 'Iter_GradSign']

    file = xlwt.Workbook(encoding="utf-8")
    # file_real_number = xlwt.Workbook(encoding = "utf-8")
    accuracy = file.add_sheet("Accuracy base line")
    accuracy.write(0, 1, "Loss")
    accuracy.write(0, 2, "Accuracy")
    adv_result_dict = {key: [] for key in adv_list}
    adv_result_cross_dict = {key: [] for key in adv_list}
    for i, model_name in enumerate(model_dict.items()):
        model = keras.models.load_model(model_name + ".h5")
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
    # file.save("report-final-6-14-2-1.xls")