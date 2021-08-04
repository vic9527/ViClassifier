def pretrained(arch, pretrained=True):
    from torchvision import models

    # if arch == 'alexnet':
    #     model = models.alexnet(pretrained=pretrained)
    # elif arch == 'vgg11':
    #     model = models.vgg11(pretrained=pretrained)
    # elif arch == 'vgg11_bn':
    #     model = models.vgg11_bn(pretrained=pretrained)
    # elif arch == 'vgg13':
    #     model = models.vgg13(pretrained=pretrained)
    # elif arch == 'vgg13_bn':
    #     model = models.vgg13_bn(pretrained=pretrained)
    # elif arch == 'vgg16':
    #     model = models.vgg16(pretrained=pretrained)
    # elif arch == 'vgg16_bn':
    #     model = models.vgg16_bn(pretrained=pretrained)
    # elif arch == 'vgg19':
    #     model = models.vgg19(pretrained=pretrained)
    # elif arch == 'vgg19_bn':
    #     model = models.vgg19_bn(pretrained=pretrained)
    # elif arch == 'resnet18':
    #     model = models.resnet18(pretrained=pretrained)
    # elif arch == 'resnet34':
    #     model = models.resnet34(pretrained=pretrained)
    # elif arch == 'resnet50':
    #     model = models.resnet50(pretrained=pretrained)
    # elif arch == 'resnet101':
    #     model = models.resnet101(pretrained=pretrained)
    # elif arch == 'resnet152':
    #     model = models.resnet152(pretrained=pretrained)
    # elif arch == 'squeezenet1_0':
    #     model = models.squeezenet1_0(pretrained=pretrained)
    # elif arch == 'squeezenet1_1':
    #     model = models.squeezenet1_1(pretrained=pretrained)
    # elif arch == 'densenet121':
    #     model = models.densenet121(pretrained=pretrained)
    # elif arch == 'densenet169':
    #     model = models.densenet169(pretrained=pretrained)
    # elif arch == 'densenet161':
    #     model = models.densenet161(pretrained=pretrained)
    # elif arch == 'densenet201':
    #     model = models.densenet201(pretrained=pretrained)
    # elif arch == 'inception_v3':
    #     model = models.inception_v3(pretrained=pretrained)
    # elif arch == 'mobilenet_v2':
    #     model = models.mobilenet_v2(pretrained=pretrained)
    # elif arch == 'shufflenet_v2_x0_5':
    #     model = models.shufflenet_v2_x0_5(pretrained=pretrained)

    from viclassifier import cfgs

    if arch in cfgs.archs:
        model = eval('models.' + arch + '(pretrained=' + str(pretrained) + ')')
        # print(model)
    else:
        raise ValueError('Unexpected network architecture', arch)
    return model


