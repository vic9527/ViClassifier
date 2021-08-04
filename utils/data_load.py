def loadFolderData(dataDir, type='train', batch_size=64):
    import torch
    from torchvision import datasets
    from viclassifier.utils import trans_gen

    data_trans = trans_gen.genTrans(type)
    data_folder = datasets.ImageFolder(dataDir, data_trans)
    class_to_idx = data_folder.class_to_idx
    data_loaders = torch.utils.data.DataLoader(data_folder, batch_size=batch_size, shuffle=True)
    return class_to_idx, data_loaders

# def get_dataloaders_imbalanced(data_dir, batch_size, sample=True, num=1):
#     from torchvision import datasets, transforms
#
#     from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
#     from collections import Counter
#     import math
#
#     # Define your transforms for the training, validation, and testing sets
#     train_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                            transforms.RandomResizedCrop(224),
#                                            transforms.RandomHorizontalFlip(),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406],
#                                                                 [0.229, 0.224, 0.225])])
#     print('Transformers generated')
#
#     train_full = datasets.ImageFolder(data_dir, transform=train_transforms)
#
#     train_set, val_set = random_split(train_full,
#                                       [math.floor(len(train_full) * 0.9), math.ceil(len(train_full) * 0.1)])
#     class_to_idx = train_set.dataset.class_to_idx
#     train_classes = [label for _, label in train_set]
#     if sample:
#         # Need to get weight for every image in the dataset
#         class_count = Counter(train_classes)
#         # class_weights = torch.Tensor(
#         #     [len(train_classes) / c for c in pd.Series(class_count).sort_index().values])
#         class_count_dict = dict(class_count)
#         for k, v in class_count_dict.items():
#             class_count_dict[k]=len(train_classes)/v
#
#         # Can't iterate over class_count because dictionary is unordered
#         sample_weights = [0] * len(train_set)
#         for idx, (image, label) in enumerate(train_set):
#             # class_weight = class_weights[label]
#             # sample_weights[idx] = class_weight
#             sample_weights[idx] = class_count_dict[label]
#
#         sampler = WeightedRandomSampler(weights=sample_weights,
#                                         num_samples=num*len(train_set), replacement=True)
#         train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
#     else:
#         # 核心部分为实际使用时替换下变量把sampler传递给DataLoader即可，注意使用了sampler就不能使用shuffle，另外需要指定采样点个数。
#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size)
#
#     print('dataloader completed!')
#
#     return class_to_idx, train_loader, val_loader


def loadFolderData_ALl(dataDir, type='train', batch_size=64, rate = .8, balance_sample=True, times = 1):
    from torchvision import datasets
    from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
    from collections import Counter
    from viclassifier.utils import trans_gen

    # import tqdm, math
    #import pandas as pd

    train_full = datasets.ImageFolder(dataDir, transform=trans_gen.genTrans(type))
    class_to_idx = train_full.class_to_idx

    # train_set, val_set = random_split(train_full,
    #                                   [math.floor(len(train_full) * rate), math.ceil(len(train_full) * (1-rate))])
    nums = len(train_full)
    pos_nums = round(nums * rate)
    train_set, val_set = random_split(train_full, [pos_nums, nums-pos_nums])

    train_classes = [label for _, label in train_set]
    if balance_sample:
        # Need to get weight for every image in the dataset
        class_count = Counter(train_classes)
        # class_weights = torch.Tensor(
        #     [len(train_classes) / c for c in pd.Series(class_count).sort_index().values])
        class_count_dict = dict(class_count)
        for k, v in class_count_dict.items():
            class_count_dict[k]=len(train_classes)/v

        # Can't iterate over class_count because dictionary is unordered
        sample_weights = [0] * len(train_set)
        for idx, (image, label) in enumerate(train_set):
            # class_weight = class_weights[label]
            # sample_weights[idx] = class_weight
            sample_weights[idx] = class_count_dict[label]

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=nums*times, replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    else:
        # 核心部分为实际使用时替换下变量把sampler传递给DataLoader即可，注意使用了sampler就不能使用shuffle，另外需要指定采样点个数。
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    if len(val_loader) == 0:
        val_loader = None
    return class_to_idx, train_loader, val_loader


def loadTxtData(txt_path, root_dir='', data_type='train', batch_size=64):
    from torch.utils.data import DataLoader
    from viclassifier.utils import txt_reader

    data = txt_reader.ImageTxt(txt_path, root_dir, data_type)
    class_to_idx = data.class_to_idx

    if data_type == 'train':
        data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    else:
        data_loader = DataLoader(dataset=data, batch_size=batch_size)

    return class_to_idx, data_loader



if __name__ == '__main__':
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    # train_dir = r''
    # valid_dir = r''
    # test_dir = r''

    # _, train_dataloaders = loadFolderData(train_dir)
    # _, validation_dataloaders = loadFolderData(train_dir, 'test')
    # _, test_dataloaders = loadFolderData(train_dir, 'test')

    # _, train_dataloaders, validation_dataloaders = loadFolderData_ALl(train_dir)

    root_dir = r'C:\Users\xxx\cls\full'
    train_txt = r'C:\Users\xxx\cls\full\full_train.txt'
    test_txt = r'C:\Users\xxx\cls\full\full_test.txt'

    train_class_to_idx, train_loader = loadTxtData(train_txt, root_dir, 'train')
    test_class_to_idx, test_loader = loadTxtData(test_txt, root_dir, 'test')

    if train_class_to_idx is not None:
        print(train_class_to_idx)
    if test_class_to_idx is not None:
        print(test_class_to_idx)

    for inputs, labels in train_loader:
        print(inputs, labels)
        break
    for inputs, labels in test_loader:
        print(inputs, labels)
        break







