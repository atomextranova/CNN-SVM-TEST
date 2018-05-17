import sys
import os
import subprocess

def attack():
    if len(sys.argv) != 2:
        # print("arg error, " + sys.argv[0] + " input_dir");
        exit(1)
    input_dir = sys.argv[1];

    if os.path.isfile(input_dir):
        exit(1);
    else:
        # for root, _, files in os.walk(input_dir):
        #     # model_name_list = ["cifar10_ResNetSVM20v3_model.121.0.2.L1.0.001", "cifar10_ResNetSVM20v3_model.170.0.1.L1.0.001"]
        #     for model_name in files:
        #         model_dir = os.path.join(root, model_name)
        #         model = keras.models.load_model(model_dir)
        #         generate_adv(model, model_name)

        for root, _, files in os.walk(input_dir):
            model_files = []
            for file in files:
                if os.path.splitext(file)[1] == ".h5":
                    model_files.append(file)
            for model_name in model_files:
                model_dir = os.path.join(root, model_name)
                try:
                    code = subprocess.run(args=['python', 'generate_adv_utils.py', model_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                except subprocess.CalledProcessError as err:
                    print("Error happens: " + err)
                else:
                    print("Exited with code: ", code)


if __name__ == "__main__":
    attack()