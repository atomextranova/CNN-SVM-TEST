
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import glob
import sys

data_augmentation = True
batch_size = 32
epochs = 100
depth = 20
version = 3
num_classes = 10

subtract_pixel_mean = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 110:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    # elif epoch > 220:
    #     lr *= 1e-5
    # elif epoch > 260:
    #     lr *= 1e-6
    # elif epoch > 300:
    #     lr *= 1e-7
    # elif epoch > 330:
    #     lr *= 1e-8
    print('Learning rate: ', lr)
    return lr

def resnet_svm(reg_l1, reg_l2, num_classes=10):
    base_model = keras.models.load_model("cifar10_ResNet20v1_model.194.h5")
    base_model.layers.pop()
    x = base_model.layers[-1].output
    x = Dense(num_classes, activation='linear', name='SVM_L1_{!s}_L2_{!s}'.format(reg_l1, reg_l2), kernel_regularizer=keras.regularizers.l1_l2(reg_l1, reg_l2))(x)
    model1 = Model(input=base_model.input, output=x)
    return model1, 'ResNetSVM%dv%d' % (depth, version)

def resnet(reg_l1, reg_l2, num_classes=10):
    base_model = keras.models.load_model("cifar10_ResNet20v1_model.194.h5")
    base_model.layers.pop()
    x = base_model.layers[-1].output
    x = Dense(num_classes, activation='softmax', name='Softmax_L1_{!s}_L2_{!s}'.format(reg_l1, reg_l2), kernel_regularizer=keras.regularizers.l1_l2(reg_l1, reg_l2))(x)
    model1 = Model(input=base_model.input, output=x)
    return model1, 'ResNet%dv%d' % (depth, version)

def resnet_ensemble(input_layer):
    model_path = sys.argv[1]
    model_list = []
    for root, _, file in os.walk(model_path):
        temp_model = keras.models.load_model(os.path.join(root, file))
        temp_model.pop(0)
        new_output = temp_model(input_layer)
        new_model = new_output
        model_list.append(new_model)
    final_output = keras.layers.average(model_list)
    ensemble_model = Model(inputs=input_layer, outputs=final_output, name='ensemble')
    return ensemble_model, 'Ensemble_ResNet%dv%d' % (depth, version)

def generate_model(l1, l2, type=None):
    if type == 'svm':
        model, model_type = resnet_svm(reg_l1=l1, reg_l2=l2)
        model.compile(loss='categorical_hinge',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
    elif type == 'ensemble':
        new_input = Input(input_shape)
        model, model_type = resnet_ensemble(new_input)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
    else:
        model, model_type = resnet(reg_l1=l1, reg_l2=l2)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models_final')
    model_name = 'cifar10_%s_model.{epoch:03d}.L1_%s.L2_%s.h5' % (model_type, str(l1), str(l2))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    board = keras.callbacks.TensorBoard(log_dir='./logs/'+str(l1)+'/'+str(l2), histogram_freq=0, batch_size=32, write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, board]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    file.write(model_name + "\n")
    file.write('Test loss:' + str(scores[0]) + "\n")
    file.write('Test accuracy:' + str(scores[1]) + "\n")

file = open("result", "w")
# regulization
# = [0.25, 0.3, 0.35, 0.4, 0.15]
l1_list = [0]
l2_list = [0]
for reg_l1 in l1_list:
    for reg_l2 in range(1, 30):
        generate_model(reg_l1, reg_l2)


# l1_list = [0]
# l2_list = [0.5, 25]
# for reg_l1 in l1_list:
#     for reg_l2 in l2_list:
#         generate_model(reg_l1, reg_l2, 'svm')

file.close()
