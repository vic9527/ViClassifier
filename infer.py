def load_model(model_path, device_type='cuda'):
    import torch
    from viclassifier.utils import dev_opt

    device = dev_opt.usingDevice(device_type)
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()
    return model


def predict(model, image_path, idx_to_class, is_show=False, device_type='cuda'):
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from viclassifier.utils import dev_opt
    from viclassifier.utils import trans_gen

    device = dev_opt.usingDevice(device_type)
    model.eval().to(device)

    transform = trans_gen.genTrans('test')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    # pytorch中的view 功能类似于numpy中的resize() 函数 把原先tensor中的数据按照行优先的顺序排成你要的形状
    # 注意原来的tensor与新的tensor是共享内存的，也就是说对其中的一个tensor进行更改的话，另外一个tensor也会自动进行相应的修改。
    # 应该使用clone()函数克隆和再进行view()，而且使⽤clone还有⼀个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。
    image_tensor_view = image_tensor.view(1, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(image_tensor_view)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
    # print("Prediction : ", idx_to_class[topclass.cpu().numpy()[0][0]],
    #       ", Score: ", topk.cpu().numpy()[0][0])


    if is_show:
        text = idx_to_class[topclass.cpu().numpy()[0][0]] + " " + str(topk.cpu().numpy()[0][0])
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('arial.ttf', 36)
        draw.text((0, 0), text, (255, 0, 0), font=font)
        image.show()

    return idx_to_class[topclass.cpu().numpy()[0][0]], topk.cpu().numpy()[0][0]


if __name__ == "__main__":
    import os, sys

    viclassifier_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(viclassifier_dir)
    sys.path.append(viclassifier_dir)

    model = load_model('D:\\myai\\projects\\tmp\\git\\viclassifier\\tmps\\model.pth')
    print(model)

    image_path = r'C:\xxx\xxx.jpg'

    # ### python字典键值对互换###
    # d1 = {'a': 1, 'b': 2, 'c': 3}
    # # 用遍历互换键值对
    # d2 = {}
    # for key, value in d1.items():
    #     d2[value] = key
    #
    # # 用列表生成器
    # d2 = {k: v for v, k in d1.items()}
    #
    # # 用zip运算符
    # d2 = dict(zip(d1.value(), d1.key()))

    class_to_idx = {'bad': 0, 'good': 1}
    idx_to_class = {k: v for v, k in class_to_idx.items()}

    predict(model, image_path, idx_to_class, is_show=False, device_type='cuda')
