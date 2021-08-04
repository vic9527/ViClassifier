# _*_ coding:utf-8 _*_

"""
Pytorch(笔记9)--读取自定义数据
https://blog.csdn.net/haiqiang1995/article/details/90348966

Pytorch源码（一）—— 简析torchvision的ImageFolder
https://www.jianshu.com/p/5bb684c4c9fc

Pytorch-ImageFolder/自定义类 读取图片数据
https://jianzhuwang.blog.csdn.net/article/details/103776245
"""


from torch.utils.data import Dataset


def imageLoader(path, type='pil'):
    # 自定义图片图片读取方式，可以自行增加resize、数据增强等操作
    if path is None:
        return
    if type == 'pil':
        from PIL import Image
        # print(path)
        return Image.open(path).convert('RGB')
    elif type == 'cv2':
        import cv2
        import numpy as np
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)


# class readTxt(Dataset):
#     # 构造函数设置默认参数
#     def __init__(self, txt_path, dir_type=None, read_type='pil'):
#         with open(txt_path, 'r') as f:
#             imgs = []
#             for line in f:
#                 line = line.strip('\n')  # 移除字符串首尾的换行符
#                 line = line.rstrip()  # 删除末尾空
#                 words = line.split()  # 以空格为分隔符 将字符串分成
#                 imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
#         self.imgs = imgs
#         self.transform = trans_gen.genTrans
#         self.mode = dir_type
#         self.type = read_type
#         self.loader = imageLoader
#
#     def __getitem__(self, index):
#         img, label = self.imgs[index]
#         # 调用定义的loader方法
#         image = self.loader(img, self.type)
#         if self.mode is not None:
#             image = self.transform(self.mode)(image)
#
#         return image, label
#
#     def __len__(self):
#         return len(self.imgs)



# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')
#
#
# def accimage_loader(path):
#     import accimage
#     # https://github.com/pytorch/accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)
#
#
# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)
#
# def find_classes(dir):
#     # 得到指定目录下的所有文件，并将其名字和指定目录的路径合并
#     # 以数组的形式存在classes中
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     # 使用sort()进行简单的排序
#     classes.sort()
#     # 将其保存的路径排序后简单地映射到 0 ~ [ len(classes)-1] 的数字上
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     # 返回存放路径的数组和存放其映射后的序号的数组
#     return classes, class_to_idx
#
# def has_file_allowed_extension(filename, extensions):
#     """Checks if a file is an allowed extension.
#
#     Args:
#         filename (string): path to a file
#
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     # 将文件的名变成小写
#     filename_lower = filename.lower()
#
#     # endswith() 方法用于判断字符串是否以指定后缀结尾
#     # 如果以指定后缀结尾返回True，否则返回False
#     return any(filename_lower.endswith(ext) for ext in extensions)
#
# def make_dataset(dir, class_to_idx, extensions):
#     images = []
#     # expanduser把path中包含的"~"和"~user"转换成用户目录
#     # 主要还是在Linux之类的系统中使用，在不包含"~"和"~user"时
#     # dir不变
#     dir = os.path.expanduser(dir)
#     # 排序后按顺序通过for循环dir路径下的所有文件名
#     for target in sorted(os.listdir(dir)):
#         # 将路径拼合
#         d = os.path.join(dir, target)
#         # 如果拼接后不是文件目录，则跳出这次循环
#         if not os.path.isdir(d):
#             continue
#         # os.walk(d) 返回的fnames是当前d目录下所有的文件名
#         # 注意：第一个for其实就只循环一次，返回的fnames 是一个数组
#         for root, _, fnames in sorted(os.walk(d)):
#             # 循环每一个文件名
#             for fname in sorted(fnames):
#                 # 文件的后缀名是否符合给定
#                 if has_file_allowed_extension(fname, extensions):
#                     # 组合路径
#                     path = os.path.join(root, fname)
#                     # 将组合后的路径和该文件位于哪一个序号的文件夹下的序号
#                     # 组成元祖
#                     item = (path, class_to_idx[target])
#                     # 将其存入数组中
#                     images.append(item)
#
#     return images

def make_dataset(dir, lines, class_to_idx, extensions=None, is_valid_file=None):
    import os
    images = []
    # dir = os.path.expanduser(dir)
    # if not ((extensions is None) ^ (is_valid_file is None)):
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    # if extensions is not None:
    #     def is_valid_file(x):
    #         return has_file_allowed_extension(x, extensions)
    # for target in sorted(class_to_idx.keys()):
    #     d = os.path.join(dir, target)
    #     if not os.path.isdir(d):
    #         continue
    #     for root, _, fnames in sorted(os.walk(d)):
    #         for fname in sorted(fnames):
    #             path = os.path.join(root, fname)
    #             if is_valid_file(path):
    #                 item = (path, class_to_idx[target])
    #                 images.append(item)
    for line in lines:
        # print(line)
        name = line.strip('\n').rstrip().split()[0]
        target = line.strip('\n').rstrip().split()[-1]
        path = os.path.join(dir, target, name)
        if is_image_file(path):
            item = (path, class_to_idx[target])
            images.append(item)

    return images


class ImageTxt(Dataset):
    def __init__(self, txt, root='', data_type=None, read_type='pil',
                 loader=imageLoader, transform=None, target_transform=None,
                 extensions=None, is_valid_file=None):
        super(ImageTxt, self).__init__()
        from viclassifier.utils import trans_gen
        with open(txt) as f:
            lines = f.readlines()
        labels = [line.strip('\n').rstrip().split()[-1] for line in lines]
        classes, class_to_idx = self._find_classes(labels)
        if root == '':
            samples = [(line.strip('\n').rstrip().split()[0],
                        class_to_idx(line.strip('\n').rstrip().split()[-1])) for line in lines if is_image_file(line)]
        else:
            samples = make_dataset(root, lines, class_to_idx)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files!\n"))

        self.data_type = data_type
        self.read_type = read_type

        # self.transform = transform
        self.transform = trans_gen.genTrans
        self.target_transform = target_transform
        self.loader = loader
        # self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path, self.read_type)

        if self.transform is not None:
            # sample = self.transform(sample)
            try:
                sample = self.transform(self.data_type)(sample)
            except:
                print("error in transform")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def _find_classes(self, labels):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        # import sys
        # if sys.version_info >= (3, 5):
        #     # Faster and available in Python 3.5 and above
        #     classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # else:
        #     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        classes = list(set(labels))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


if __name__ == '__main__':
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    from torch.utils.data import DataLoader

    root_dir = r'C:\Users\xxx\cls\full'

    train_txt = r'C:\Users\xxx\cls\full\full_train.txt'
    test_txt = r'C:\Users\xxx\cls\full\full_test.txt'

    train_data = ImageTxt(train_txt, root_dir, 'train')
    test_data = ImageTxt(test_txt, root_dir, 'test')

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64)

    if train_data.class_to_idx is not None:
        print(train_data.class_to_idx)
    if test_data.class_to_idx is not None:
        print(test_data.class_to_idx)

    if train_loader is not None:
        for inputs, labels in train_loader:
            print(inputs, labels)
            break
    if test_loader is not None:
        for inputs, labels in test_loader:
            print(inputs, labels)
            break


