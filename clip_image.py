import h5py
import sys
import os
import numpy as np

file_dir = sys.argv[1]
save_dir = sys.argv[2]
file_name = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
             and file.startswith('adv')
             and file.endswith('.h5')]

# model_name = [os.path.splitext(file)[0] for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))
#              and file.startswith('cifar')
#              and file.endswith('.h5')]

for file in file_name:
    with h5py.File(os.path.join(file_dir, file)) as hf_read:
        with h5py.File(os.path.join(save_dir, file), 'w') as hf_write:
            clipped_image = np.clip(hf_read['adv'][:], 0, 1)
            hf_write.create_dataset(name='adv', data=clipped_image)

