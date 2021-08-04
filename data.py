def loader(root_dir, batch_size=64, type='folder', boolSplit=False, isbalance=False):
    import os
    from viclassifier.utils import data_load

    class_to_idx = None
    train_dataloaders = None
    test_dataloaders = None
    if type == 'folder' and not boolSplit and not isbalance:
        if 'train' in os.listdir(root_dir) or 'test' in os.listdir(root_dir):
            if 'train' in os.listdir(root_dir):
                train_dir = os.path.join(root_dir, 'train')
                class_to_idx, train_dataloaders = data_load.loadFolderData(train_dir, batch_size=batch_size)
            if 'test' in os.listdir(root_dir):
                test_dir = os.path.join(root_dir, 'test')
                _, test_dataloaders = data_load.loadFolderData(test_dir, 'test', batch_size=batch_size)
                if class_to_idx is None:
                    class_to_idx = _

        else:
            class_to_idx, train_dataloaders = data_load.loadFolderData(root_dir, batch_size=batch_size)

    elif type == 'folder' and not boolSplit and isbalance:
        class_to_idx, train_dataloaders, test_dataloaders \
            = data_load.loadFolderData_ALl(root_dir, type='train', batch_size=batch_size,
                                           rate=1, balance_sample=True, times=1)

    elif type == 'folder' and boolSplit and isbalance:
        class_to_idx, train_dataloaders, test_dataloaders \
            = data_load.loadFolderData_ALl(root_dir, type='train', batch_size=batch_size,
                                           rate=.8, balance_sample=True, times=1)

    elif type == 'txt':
        # if 'train.txt' in os.listdir(root_dir) or 'test.txt' in os.listdir(root_dir):
        #     if 'train.txt' in os.listdir(root_dir):
        #         train_txt = os.path.join(root_dir, 'train.txt')
        #         class_to_idx, train_dataloaders = data_load.loadTxtData(train_txt, root_dir, 'train', batch_size=batch_size)
        #
        #     if 'test.txt' in os.listdir(root_dir):
        #         test_txt = os.path.join(root_dir, 'test.txt')
        #         _, test_dataloaders = data_load.loadTxtData(test_txt, root_dir, 'test', batch_size=batch_size)
        #         if class_to_idx is None:
        #             class_to_idx = _

        train_txt = os.path.join(root_dir, 'train.txt')
        test_txt = os.path.join(root_dir, 'test.txt')

        if os.path.isfile(train_txt):
            class_to_idx, train_dataloaders = data_load.loadTxtData(train_txt, root_dir, 'train', batch_size=batch_size)
        else:
            print(train_txt + ' is not exist!')

        if os.path.isfile(test_txt):
            _, test_dataloaders = data_load.loadTxtData(test_txt, root_dir, 'test', batch_size=batch_size)
            if class_to_idx is None:
                class_to_idx = _
        else:
            print(test_txt + ' is not exist!')

    else:
        print('Type Input Error!')

    return class_to_idx, train_dataloaders, test_dataloaders


if __name__ == '__main__':
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    root_dir = r'C:\Users\xxx\cls\full'

    class_to_idx, train_loader, test_loader = loader(root_dir, 256,
                                                     type='txt', boolSplit=False, isbalance=False)

    print(class_to_idx)

    if train_loader is not None:
        for inputs, labels in train_loader:
            print(len(inputs), labels)
            break
    if test_loader is not None:
        for inputs, labels in test_loader:
            print(len(inputs), labels)
            break


