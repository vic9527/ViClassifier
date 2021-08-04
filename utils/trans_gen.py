"""
[PyTorch 学习笔记] 2.3 二十二种 transforms 图片数据预处理方法
https://www.cnblogs.com/zhangxiann/p/13570884.html
"""


def genTrans(type='train'):
    """
    前面的(0.485,0.456,0.406)表示均值，分别对应的是RGB三个通道；后面的(0.229,0.224,0.225)则表示的是标准差
    这上面的均值和标准差的值是ImageNet数据集计算出来的，所以很多人都使用它们。
    """
    from torchvision import transforms

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    size = 224

    if type == 'train':
        transformers = transforms.Compose([
            transforms.RandomRotation((0, 360), expand=True),
            # transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
            transforms.Normalize(mean, std)])
        return transformers

    elif type == 'test':
        transformers = transforms.Compose([
            transforms.Resize(size),
            # transforms.Resize(255),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
            transforms.Normalize(mean, std)])
        return transformers
    else:
        return "KO!"
