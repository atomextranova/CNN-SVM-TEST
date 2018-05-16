import keras
import numpy as np
import h5py
import sys
from sklearn import svm, metrics
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from keras.datasets import cifar10


def load_base_model():
    if len(sys.argv) != 2:
        print("Please provide the path to the base model")
        exit(1)
    model = keras.models.load_model(sys.argv[1])
    model.layers.pop()
    x = model.layers[-1].output
    model_extraction = keras.Model(inputs=model.inputs, outputs=x)
    model_extraction.summary()
    return model_extraction


def load_data():
    try:
        with h5py.File("data.h5", "r") as hf:
            print("Using existing data files")
            return hf['x_train_svm'][:], hf['y_train'][:], hf['x_test_svm'][:], hf['y_test'][:]
    except:

        print("Data files not found. \nGenerating new data files.")
        model = load_base_model()

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        x_train_svm = model.predict(x_train)
        x_test_svm = model.predict(x_test)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        with h5py.File("data.h5", "w") as hf:
            hf.create_dataset(name="x_train_svm", data=x_train_svm)
            hf.create_dataset(name="y_train", data=y_train)
            hf.create_dataset(name="x_test_svm", data=x_test_svm)
            hf.create_dataset(name="y_test", data=y_test)
        print("Files generated successfully")
        return x_train_svm, y_train, x_test_svm, y_test

if __name__ == "__main__":
    x_train_svm, y_train, x_test_svm, y_test = load_data()
    print("Data loading successfully.")
    scores = ['accuracy', 'precision_macro', 'recall_macro']

    tuned_parameters = [{'C': [0.1, 1, 10, 100, 1000, 10000]}]
    random_state = 12883823
    # rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    reg = 'l2'
    lin_svm = svm.LinearSVC(penalty=reg)
    for score in scores:
        print("# Tuning hyper-parameters %s\n" % score)

        clf = GridSearchCV(lin_svm, tuned_parameters, cv=10, scoring=score, error_score=0, n_jobs=4)
        clf.fit(x_train_svm, y_train.ravel())
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = clf.predict(y_train)
        print()
        print(classification_report(y_test.ravel(), y_pred))
        print()


    # joblib.dump(lin_svm, 'svm.pkl')

