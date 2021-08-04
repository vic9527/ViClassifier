archs = ['alexnet',
         'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19', 'vgg19_bn',
         'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
         'densenet121', 'densenet169', 'densenet161', 'densenet201',
         'googlenet', 'inception_v3',
         'mobilenet_v2',
         'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
         'squeezenet1_0', 'squeezenet1_1',
         'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
         
arch = 'resnet18'

data_directory = r'C:\xxx\cls\full'  # path of your dataset

# num_labels = 2  # 可有可无

num_epochs = 1000

batch_size = 64

learning_rate = 0.0001
