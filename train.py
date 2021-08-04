def load_model(model_path, device_type='cuda'):
    import torch
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()
    return model

def criterion():
    from torch import nn

    return nn.NLLLoss()


def optimization(model, learning_rate):
    from torch import optim

    # if arch == 'resnet18' or arch == 'resnet34' or \
    #         arch == 'resnet50' or arch == 'resnet101' or \
    #         arch == 'resnet152' or arch == 'inception_v3':
    #     optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    # else:
    #     optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return optim.Adam(model.parameters(), lr=learning_rate)


def eval(model, validation_dataloaders, device):
    import torch
    from tqdm import tqdm

    cost = criterion()
    validation_loss = 0
    validation_accuracy = 0
    with torch.no_grad():
        for valid_inputs, valid_labels in tqdm(validation_dataloaders, ascii=True, desc="Validation:"):
            valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)

            valid_output = model.forward(valid_inputs)

            valid_loss = cost(valid_output, valid_labels)
            print("\nValid Loss: ", valid_loss)

            validation_loss += valid_loss.item()

            ps = torch.exp(valid_output)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == valid_labels.view(*top_class.shape)
            # torch.FloatTensor 可运行CPU和GPU，torch.cuda.FloatTensor 只能运行GPU。
            # 可用numpy计算，缺点是如果是在GPU上需要先转到CPU上再转换成numpy格式。
            # import numpy
            # print(numpy.mean(equals.cpu().numpy() + 0))
            # print(numpy.mean(equals.cpu().numpy().astype(int)))
            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return validation_loss / len(validation_dataloaders), validation_accuracy / len(validation_dataloaders)


def run(model, train_dataloaders, validation_dataloaders=None,
        save_path='./tmps/model.pth', eval_rate=1,
        epochs=10, learning_rate=0.001, device_type='cuda'):
    import torch
    from tqdm import tqdm
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model.to(device)

    cost = criterion()
    optimizer = optimization(model, learning_rate)

    print("Start Training!")
    bestSave = 0
    best_e = 0
    for e in range(epochs):
        e = e + 1
        # print(e)
        train_loss = 0
        train_accuracy = 0

        # d = {'loss':0.2,'learn':0.8}
        # for i in tqdm(range(50),desc='进行中',ncols=10,postfix=d):
        # #desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息

        model.train()

        for inputs, labels in tqdm(train_dataloaders, ascii=True, desc="Train-" + str(e)):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            logOutput = model.forward(inputs)

            iter_loss = cost(logOutput, labels)
            train_loss += iter_loss.item()
            print("\nIter Loss: ", iter_loss)

            # # for nets that have multiple outputs such as inception
            # if isinstance(outputs, tuple):
            #     loss = sum((criterion(o, labels) for o in outputs))
            # else:
            #     loss = criterion(outputs, labels)

            ps = torch.exp(logOutput)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            iter_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            train_accuracy += iter_accuracy
            # print("Iter Accuracy：", iter_accuracy)
            # print("BestSave Accuracy：", bestSave)

            # 反向传播
            iter_loss.backward()

            # 参数更新
            optimizer.step()

        t_loss, t_acc = train_loss / len(train_dataloaders), train_accuracy / len(train_dataloaders)
        print("Epoch Loss {:.3f}, Epoch Accuracy: {:.3f}".format(t_loss, t_acc))

        # else:
        if validation_dataloaders is not None:
            if e % eval_rate == 0:
                model.eval()
                torch.save(model, save_path)
                # t_loss, t_acc = eval(train_dataloaders)
                # print("Train Loss {:.3f}, Train Accuracy: {:.3f}".format(t_loss, t_acc))
                v_loss, v_acc = eval(model, validation_dataloaders, device)
                print("Validation Loss {:.3f}, Validation Accuracy: {:.3f}".format(v_loss, v_acc))

                if v_acc > bestSave:
                    import os

                    # if v_loss < bestSave:
                    torch.save(model, os.path.join(os.path.splitext(save_path)[0] + '-best' + os.path.splitext(save_path)[-1]))
                    print("Best Loss {:.3f}, Best Accuracy: {:.3f}".format(v_loss, v_acc))
                    bestSave = v_acc
                    # bestSave = v_loss
                    best_e = e

        print("BestSave Accuracy：{:.3f} When Epoch-{}".format(bestSave, best_e))

    model.eval()
    torch.save(model, save_path)
    print("End Training!")


if __name__ == "__main__":
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    from viclassifier import args
    from viclassifier import data

    # 使用默认参数
    parser = args.get_arguments()
    parms = parser.parse_args()  # 测试无法使用，运行该文件时使用。

    # root_dir = r'C:\Users\xxx\cls\full'
    # parms.data_directory = root_dir
    class_to_idx, train_loader, test_loader = data.loader(parms.data_directory, parms.batch_size,
                                                          type='txt', boolSplit=False, isbalance=False)

    # model = load_model('C:\\xxx\\viclassifier\\tmps\\model.pth')
    # print(model)


    if class_to_idx is not None:
        print(class_to_idx, len(class_to_idx))
        if train_loader is not None:
            from viclassifier.models import transfer

            model = transfer.model(parms.arch, len(class_to_idx))
            print(model)
            print("load transfer model with pre-trained_model: %s" % parms.arch)

            run(model, train_loader, test_loader)
