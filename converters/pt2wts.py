def convert(pth_path, wts_path, device_type='cuda'):
    import struct
    import torch
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model = torch.load(pth_path, map_location=device)
    model.to(device)
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()

    # print('model: ', model)
    # print('state dict: ', model.state_dict().keys())

    # # 生成数据测试
    # tmp = torch.ones(1, 3, 224, 224).to(device)
    # print('input: ', tmp)
    # out = model(tmp)
    # print('output:', out)

    f = open(wts_path, 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        # print('key: ', k)
        # print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()  #在CPU上执行
        f.write("{} {}".format(k, len(vr)))
        print("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            # print(" ")
            f.write(struct.pack(">f", float(vv)).hex())
            # print(struct.pack(">f", float(vv)).hex())
        f.write("\n")
        # print("\n")


if __name__ == "__main__":
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print('viclassifier_dir:', viclassifier_dir)
    sys.path.append(viclassifier_dir)

    pth_path = r'../examples/model.pth'
    wts_path = r'../examples/model.wts'
    convert(pth_path, wts_path)
