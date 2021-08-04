def model(arch, num_labels=2, pretrained=True):
    from torch import nn
    import viclassifier.models.model as tvmodel

    model = tvmodel.pretrained(arch, pretrained)

    # for i, param in enumerate(model.classifier.parameters()):
    #     # print(i, param)
    #     param.requires_grad = True

    for param in model.parameters():
        # param.requires_grad = False
        param.requires_grad = True

    num_filters = None

    if arch == 'resnet18' or arch == 'resnet34' or arch == 'resnet50' or arch == 'resnet101' or arch == 'resnet152' or\
            arch == 'inception_v3' or arch == 'shufflenet_v2_x0_5':
        num_filters = model.fc.in_features

        if num_filters is None:
            print('num_filters is error!')

        # fc = nn.Sequential(nn.Linear(num_filters, 512),
        #                    nn.ReLU(),
        #                    nn.Dropout(0.2),
        #                    nn.Linear(512, 256),
        #                    nn.ReLU(),
        #                    nn.Dropout(0.2),
        #                    nn.Linear(256, num_labels),
        #                    nn.LogSoftmax(dim=1))

        fc = nn.Sequential(nn.Linear(num_filters, num_labels),
                           nn.LogSoftmax(dim=1))
        model.fc = fc
    else:
        if type(model.classifier) is nn.modules.Sequential:
            for classif in model.classifier:
                try:
                    num_filters = classif.in_features
                    break
                except AttributeError:
                    continue
        elif type(model.classifier) is nn.modules.Linear:
            num_filters = model.classifier.in_features

        if num_filters is None:
            print('num_filters is error!')

        # classifier = nn.Sequential(nn.Linear(num_filters, 512),
        #                            nn.ReLU(),
        #                            nn.Dropout(0.2),
        #                            nn.Linear(512, 256),
        #                            nn.ReLU(),
        #                            nn.Dropout(0.2),
        #                            nn.Linear(256, num_labels),
        #                            nn.LogSoftmax(dim=1))

        classifier = nn.Sequential(nn.Linear(num_filters, num_labels),
                                   nn.LogSoftmax(dim=1))

        if arch == 'mobilenet_v2':
            classifier = nn.Sequential(nn.Dropout(0.2),
                                       nn.Linear(num_filters, num_labels),
                                       nn.LogSoftmax(dim=1))
        model.classifier = classifier

    # print(num_filters, num_labels)

    return model
