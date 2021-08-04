def usingDevice(device_type='cuda'):
    import torch

    # device = torch.device("cuda" if device_type == 'cuda' and torch.cuda.is_available() else "cpu")

    '''
    str_1 = "123"
    str_2 = "Abc"
    str_3 = "123Abc"

    #用isdigit函数判断是否数字
    print(str_1.isdigit())
    Ture
    print(str_2.isdigit())
    False
    print(str_3.isdigit())
    False

    #用isalpha判断是否字母
    print(str_1.isalpha())
    False
    print(str_2.isalpha())
    Ture
    print(str_3.isalpha())
    False

    #isalnum判断是否数字和字母的组合
    print(str_1.isalnum())
    Ture
    print(str_2.isalnum())
    Ture
    print(str_1.isalnum())
    Ture注意：如果字符串中含有除了字母或者数字之外的字符，比如空格，也会返回False
    '''
    if torch.cuda.is_available():

        if device_type == 'cuda':
            device = torch.device(device_type)
        elif device_type.isdigit() and int(device_type) >= 0:
            device = torch.device("cuda:{}".format(device_type))
        else:
            device = torch.device("cpu")
    else:

        device = torch.device("cpu")

    print("using device type with: ", device)

    return device