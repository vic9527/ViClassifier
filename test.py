def load_model(model_path, device_type='cuda'):
    import torch
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()
    return model


def run(model, test_dataloaders, device_type='cuda'):
    import torch
    from tqdm import tqdm
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model.eval().to(device)
    test_accuracy = 0
    equals_all = torch.Tensor().to(device)
    with torch.no_grad():
        for test_inputs, test_labels in tqdm(test_dataloaders, ascii=True, desc="Test:"):
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_output = model.forward(test_inputs)

            ps = torch.exp(test_output)

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == test_labels.view(*top_class.shape)
            equals_float = equals.type(torch.FloatTensor).to(device)
            equals_all = torch.cat((equals_all, equals_float), 0)  # 在 0 维(纵向)进行拼接，在 1 维(横向)进行拼接。

            # iter_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            iter_accuracy = torch.mean(equals_float).item()

            print("\nIter Accuracy: {:.3f}".format(iter_accuracy))

            test_accuracy += iter_accuracy

        all_accuracy = torch.mean(equals_all.type(torch.FloatTensor)).item()

    # print("Test Accuracy: {:.3f}".format(test_accuracy / len(test_dataloaders)))
    print("Test Accuracy: {:.3f}".format(all_accuracy))
    # return test_accuracy / len(test_dataloaders)
    return all_accuracy


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

    class_to_idx, train_loader, test_loader = data.loader(parms.data_directory, parms.batch_size,
                                                          type='txt', boolSplit=False, isbalance=False)

    model = load_model('C:\\xxx\\viclassifier\\tmps\\model.pth')
    print(model)

    print(class_to_idx)
    if test_loader is not None:
        run(model, test_loader)
