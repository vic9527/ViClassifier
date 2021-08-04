"""
如何检查python模块是否存在而不导入它
https://www.pythonheidong.com/blog/article/51529/e19bb3539880b1d0cdaf/

# import os, sys
# import importlib
# spec = importlib.util.find_spec("dev_opt")
# found = spec is not None

# # 将当前工作目录加入path
# # sys.path.append:添加环境变量
# # os.getcwd：返回当前工作目录（注意是工作目录cwd哦）
# # os.chdir(dir)：返回修改的dir路径
# sys.path.append(os.getcwd())
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

# # 将当前文件的上级目录添加到path
# # sys.path.append:添加环境变量
# # os.path.dirname：返回路径名的目录部门（可以获取多次）
# # os.path.abspath：获取文件绝对路径  os.path.abspath(__file__) 直接写成__file__也可以
# os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# env_dist = os.environ # environ是在os.py中定义的一个dict environ = {}
# print(env_dist['PYTHONPATH'])
# print(os.path, os.sys.path, sys.path)

"""


class VicModel(object):
    def __init__(self, arch='densenet121', label=2, is_custom=False, show_model=False, device_type='cuda'):
        from viclassifier.utils import dev_opt

        self.device_type = device_type
        self.device = dev_opt.usingDevice(self.device_type)

        if arch.endswith('.pt') or arch.endswith('.pth') or arch.endswith('.pkl'):
            self.model = self.load_model(arch)
            print("load model success with: %s" % arch)
            if show_model:
                print(self.model)

        else:
            from viclassifier import cfgs
            if arch in cfgs.archs:
                if is_custom:
                    from viclassifier.models import custom
                    self.model = custom.model()
                else:
                    from viclassifier.models import transfer
                    self.arch = arch
                    self.label = label
                    self.model = transfer.model(self.arch, self.label)
                    print("load transfer model with pre-trained_model: %s" % self.arch)
                if show_model:
                    print(self.model)

            else:
                raise ValueError('Unexpected input parameters', arch)

    def load_model(self, model_path, device_type=None):
        import torch

        device = self.device
        if device_type is not None:
            from viclassifier.utils import dev_opt
            device = dev_opt.usingDevice(device_type)
        model = torch.load(model_path, map_location=device)
        model.to(device)
        # 测试时不启用 BatchNormalization 和 Dropout
        model.eval()
        return model

    def load_weight(self, model_path, device_type=None):
        import torch

        device = self.device
        if device_type is not None:
            from viclassifier.utils import dev_opt
            device = dev_opt.usingDevice(device_type)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        # 测试时不启用 BatchNormalization 和 Dropout
        self.model.eval()
        return self.model

    def train_data(self, train_dataloaders, validation_dataloaders=None,
                   save_path='./tmps/model.pth', eval_rate=1,
                   epochs=10, learning_rate=0.001, device_type=None):
        if device_type is not None:
            device_type = device_type
        else:
            device_type = self.device_type
        from viclassifier import train
        train.run(self.model, train_dataloaders, validation_dataloaders,
                  save_path, eval_rate,
                  epochs, learning_rate, device_type)

    def test_data(self, test_dataloaders, device_type=None):
        if device_type is not None:
            device_type = device_type
        else:
            device_type = self.device_type
        from viclassifier import test
        test.run(self.model, test_dataloaders, device_type)

    def infer_image(self, image_path, idx_to_class, is_show=False, device_type=None):
        if device_type is not None:
            device_type = device_type
        else:
            device_type = self.device_type
        from viclassifier import infer
        return infer.predict(self.model, image_path, idx_to_class, is_show, device_type)

    def save_model(self, save_path, mode='full'):
        import os
        import torch

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if mode == 'full':
            torch.save(self.model, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def save_checkpoint(self, save_path, class_to_idx):
        import os
        import torch

        self.model.class_to_idx = class_to_idx
        if self.arch == 'resnet18' or self.arch == 'resnet34' or self.arch == 'resnet50' or self.arch == 'resnet101' or self.arch == 'resnet152' or self.arch == 'inception_v3':
            checkpoint = {'structure': self.arch,
                          'hidden_layer1': 512,
                          'fc': self.model.fc,
                          'state_dict': self.model.state_dict(),
                          'class_to_idx': self.model.class_to_idx}
        else:
            checkpoint = {'structure': self.arch,
                          'hidden_layer1': 512,
                          'classifier': self.model.classifier,
                          'state_dict': self.model.state_dict(),
                          'class_to_idx': self.model.class_to_idx}
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(checkpoint, save_path)


if __name__ == '__main__':
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    from viclassifier import args
    from viclassifier import data
    # 使用默认参数
    parser = args.get_arguments()
    # parms = parser.parse_args(args=[])  # 默认参数，测试使用，生产使用会导致无法命令行传参。
    # parms = parser.parse_args(['--batch_size', '128', '--learning_rate', '0.002'])
    # parms = parser.parse_args(args=['--batch_size', '256', '--learning_rate', '0.003'])
    parms = parser.parse_args()  # 测试无法使用，运行该文件时使用。
    # print(parms)

    class_to_idx, train_loader, test_loader = data.loader(parms.data_directory, parms.batch_size,
                                                          type='txt', boolSplit=False, isbalance=False)

    print('class_to_idx: ', class_to_idx)
    idx_to_class = {k: v for v, k in class_to_idx.items()}
    print('idx_to_class: ', idx_to_class)

    parms.arch = 'D:\\xxx\\viclassifier\\tmps\\model.pth'

    if class_to_idx is not None:
        model = VicModel(parms.arch, len(class_to_idx))
        # if train_loader is not None:
        #     model.train_data(train_loader, test_loader)
        # if test_loader is not None:
        #     model.test_data(test_loader)

        # image_path = r'C:\Users\MLT\Desktop\精悦蓉\cls\full\good\20210719_180953_1.jpg'

        image_dir = r'C:\Users\xxx\cls\full'
        from tqdm import tqdm
        for root, dirs, files in os.walk(image_dir):
            # print(root, dirs, files)
            flist = [os.path.join(root, f) for f in files if f.endswith('.jpg')]
        for image_path in tqdm(flist):
            print('\n' + image_path)
            print(model.infer_image(image_path, idx_to_class))
