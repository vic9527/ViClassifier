import os, sys

viclassifier_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('viclassifier_dir:', viclassifier_dir)
sys.path.append(viclassifier_dir)

data_directory = sys.argv[1]
batch_size = int(sys.argv[2])
arch = sys.argv[3]
save_path = sys.argv[4]
eval_rate = int(sys.argv[5])
epochs = int(sys.argv[6])
learning_rate = float(sys.argv[7])
device_type = sys.argv[8]
print('data_directory:', data_directory)
print('batch_size:', batch_size)
print('arch:', arch)
print('save_path:', save_path)
print('eval_rate:', eval_rate)
print('epochs:', epochs)
print('learning_rate:', learning_rate)
print('device_type:', device_type)

import viclassifier as vic

class_to_idx, train_loader, test_loader = vic.data.loader(data_directory, batch_size,
                                                      type='txt', boolSplit=False, isbalance=False)

print('class_to_idx: ', class_to_idx)
idx_to_class = {k: v for v, k in class_to_idx.items()}
print('idx_to_class: ', idx_to_class)

if class_to_idx is not None:
    model = vic.main.VicModel(arch, len(class_to_idx), device_type=device_type)
    if train_loader is not None:
        model.train_data(train_loader, test_loader, save_path, eval_rate, epochs, learning_rate)
    if test_loader is not None:
        model.test_data(test_loader)
