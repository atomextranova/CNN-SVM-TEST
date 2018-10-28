import argparse
import multiprocessing
import os
import threading
import time

import foolbox
import h5py
import keras
import numpy as np
from keras.datasets import cifar10

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


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

def clip_image(image):
    return np.clip(image, 0, 1)

def utils_adv(attacker, img, label):
        adv_image = attack(img, label)
        if adv_image is not None:
            return adv_image
        else:
            print("Fail to generate adv image. Appending original image.\n")
            return label

def attack_wrapper(save_dir, model_name, attack, name, gap, lock, part=False):
    orig_image, orig_label, mean_of_image = read_orig(gap)
    save_base_path = os.path.join(save_dir, model_name)

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
    name += ("_" + model_name)
    adv = []
    record = open(os.path.join(save_base_path, "{}.txt".format(name)), 'w')
    print("--- {} started ---\n".format(name))
    start = time.time()
    # with ThreadPoolExecutor(max_workers=os.cpu_count()):

    for i, (img, label) in enumerate(zip(orig_image, orig_label)):
        # for i in range(1):
        # for i in range(10):
        # if i % 10 == 0:
        #     print("Generateing %d images with gap %d\n" % (i, gap))
        adv_image = attack(img, label)
        if adv_image is not None:
            adv.append(adv_image)
        else:
            print("Fail to generate adv image. Appending original image.\n")
            adv.append(orig_image[i])
    adv = np.array(adv, 'float32')
    completion_msg = "--- {} {} seconds ---\n".format(name, (time.time() - start))
    print(completion_msg)
    record.write(completion_msg)
    # record.write("Sucessfully generated %d images with gap %d, including %d original images\n" % (
    #     int(adv.shape[0]), gap, count))
    if part:
        name += "_part"
    elif gap != 1:
        name += "_gap"

    with h5py.File(os.path.join(save_base_path, "adv_{}.h5".format(name)), "w") as hf:
        clipped_adv = clip_image(adv + mean_of_image)  # add mean for standard graph, limit data range to be [0, 1]
        min_val = np.amin(np.abs(clipped_adv - orig_image - mean_of_image)) * 255
        max_val = np.amax(np.abs(clipped_adv - orig_image - mean_of_image)) * 255
        avg_val = np.sum(np.abs(clipped_adv - orig_image - mean_of_image)) / clipped_adv.size * 255
        record.write("Min: {}".format(min_val))
        record.write("Max: {}".format(max_val))
        record.write("Avg: {}".format(avg_val))
        hf.create_dataset('adv', data=clipped_adv)

    record.close()
    return

def decode_args(arg_list):
    return arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4]

def attack_group(model_adv, process_size, model_name, save_dir, lock, gap):
    # start = time.time()

    attack_deep_fool_l2 = foolbox.attacks.DeepFoolL2Attack(model_adv)

    attack_DFL_INF = foolbox.attacks.DeepFoolLinfinityAttack(model_adv)

    attack_wrapper(save_dir, model_name, attack_deep_fool_l2, "DeepFool_L_2", gap, lock)
    attack_wrapper(save_dir, model_name, attack_DFL_INF, 'DeepFool_L_INF', gap, lock)

    # attack_LBFGSAttack = foolbox.attacks.LBFGSAttack(model_adv)
    # attack_wrapper(save_dir, process_size, model_name, attack_LBFGSAttack, 'LBGFS', gap, lock)
    #
    # attack_GaussianBlur = foolbox.attacks.GaussianBlurAttack(model_adv)
    # attack_wrapper(save_dir, process_size, model_name, attack_GaussianBlur, "Gaussian_Blur", gap, lock)
    #
    attack_IterGradSign = foolbox.attacks.IterativeGradientSignAttack(model_adv)
    attack_wrapper(save_dir, process_size, model_name, attack_IterGradSign, "Iter_GradSign", gap, lock)

    attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
    attack_wrapper(save_dir, process_size, model_name, attack_IterGrad, "Iter_Grad", gap, lock)
    # # # print("--- " + str(1) + "takes %s seconds ---\n" % (time.time() - start))

def attack_group_1(model_adv, model_name, save_dir, lock, gap):
    # start = time.time()

    attack_deep_fool_l2 = foolbox.attacks.DeepFoolL2Attack(model_adv)

    attack_DFL_INF = foolbox.attacks.DeepFoolLinfinityAttack(model_adv)

    attack_wrapper(save_dir, model_name, attack_deep_fool_l2, "DeepFool_L_2", gap, lock)
    attack_wrapper(save_dir, model_name, attack_DFL_INF, 'DeepFool_L_INF', gap, lock)

    # attack_LBFGSAttack = foolbox.attacks.LBFGSAttack(model_adv)
    # attack_wrapper(save_dir, model_name, attack_LBFGSAttack, 'LBGFS', gap, lock)

    attack_GaussianBlur = foolbox.attacks.GaussianBlurAttack(model_adv)
    attack_wrapper(save_dir, model_name, attack_GaussianBlur, "Gaussian_Blur", gap, lock)

    # attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
    # attack_wrapper(save_dir, model_name, attack_IterGrad, "Iter_Grad", gap, lock)
    # # print("--- " + str(1) + "takes %s seconds ---\n" % (time.time() - start))


def attack_group_2(model_adv, model_name, save_dir, lock, gap):
    # start = time.time()
    attack_IterGradSign = foolbox.attacks.IterativeGradientSignAttack(model_adv)
    attack_wrapper(save_dir, model_name, attack_IterGradSign, "Iter_GradSign", gap, lock)
    # print("--- " + str(2) + "takes %s seconds ---\n" % (time.time() - start))


def attack_group_3(model_adv, model_name, save_dir, lock, gap):
    # start = time.time()
    attack_IterGrad = foolbox.attacks.IterativeGradientAttack(model_adv)
    attack_wrapper(save_dir, model_name, attack_IterGrad, "Iter_Grad", gap, lock)
    # print("--- " + str(3) + "takes %s seconds ---\n" % (time.time() - start))


# def attack_group_4(model_adv, model, model_name, lock):
#     # start = time.time()
#     attack_Local = foolbox.attacks.LocalSearchAttack(model_adv)
#     attack_Single_Pixel = foolbox.attacks.SinglePixelAttack(model_adv)
#     attack_wrapper(model, model_name, attack_Local, "Local_Search", gap, lock)
#     attack_wrapper(model, model_name, attack_Single_Pixel, "Single_Pixel", gap, lock)
#     # print("--- " + str(4) + "takes %s seconds ---\n" % (time.time() - start))

def attack_worker(arg_list):
    save_dir, process_size, model_name, model_dir, gap = decode_args(arg_list)
    print("{}: attack started".format(model_name))
    start = time.time()
    model = keras.models.load_model(model_dir)
    print('Successfully loaded model: {}'.format(model_name))
    # make thread ready manually
    model._make_predict_function()
    model_adv = foolbox.models.KerasModel(model, bounds=(-1, 1), preprocessing=((0, 0, 0), 1))

    # thread_list = []
    my_args_dict = dict(model_adv=model_adv, save_dir=save_dir, model_name=model_name, lock=threading.Lock(), gap=gap)
    attack_group_1(my_args_dict)
    # attack_group(model_adv, process_size, model_name, save_dir, threading.Lock(), gap)
    # attack_group_1(model_adv, model_name, save_dir, threading.Lock(), gap)  # Debug line
    # thread_list.append(threading.Thread(target=attack_group_1, kwargs=my_args_dict))
    # thread_list.append(threading.Thread(target=attack_group_2, kwargs=my_args_dict))
    # thread_list.append(threading.Thread(target=attack_group_3, kwargs=my_args_dict))
    # for thread in thread_list:
    #     thread.start()
    # for thread in thread_list:
    #     thread.join()
    print("--- " + model_name + "takes %s seconds ---\n" % (time.time() - start))


def attack(save_dir, process_size, model_names, model_dirs, gap):
    generate_orig()
    attacker_pool = multiprocessing.Pool(processes=process_size)
    args_list = [[save_dir, process_size, model_name, model_dir, gap] for model_name, model_dir in list(zip(model_names, model_dirs))]
    attacker_pool.map(attack_worker, args_list)
    # for model_name, model_dir in list(zip(model_names, model_dirs)):
    #     attack_worker([save_dir, process_size, model_name, model_dir, gap])

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print('Provide the path argument for model or model directory')
    # model_dir = sys.argv[1]

    parser = argparse.ArgumentParser("")
    parser.add_argument('-m', '--model', nargs='*',
                        help="specify all models or model directories that is to be attacked")
    parser.add_argument('-s', '--save_dir', help="specify the save directory for attack file", default=None)
    parser.add_argument('-p', '--process_size', help='number of processes', type=int, default=1)
    # parser.add_argument('-t', '--thread_num', help='number of threads for each attack', type=int, default=4)
    parser.add_argument('-g', '--gap', help='select images with gap ([::10])', type=int, default=1)
    # parser.add_argument('-v', '--verbose', help="whether print the progress or not", action='store_true')
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
    # verbose = args.verbose
    process_size = args.process_size
    gap = args.gap

    model_names = []
    model_dirs = []

    for model_loc in model_locs:
        if os.path.isfile(model_loc):
            model_names.append(os.path.splitext(model_loc)[0])
            model_dirs.append(model_loc)
        elif os.path.isdir(model_loc):
            model_files_sub =[file for file in os.listdir(model_loc) if file.startswith('ens') and file.endswith('.h5')]
            model_names_sub = [os.path.splitext(model_file)[0] for model_file in model_files_sub]
            model_dirs_sub = [os.path.join(model_loc, model_file) for model_file in model_files_sub]
            model_names.extend(model_names_sub)
            model_dirs.extend(model_dirs_sub)
        else:
            raise ValueError('In {}: Not file or directory'.format(model_loc))

    attack(save_dir, process_size, model_names, model_dirs, gap)
