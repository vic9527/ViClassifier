def writetxt(save_path, save_txt):
    import os

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, mode="a", encoding='utf-8') as f:
        f.write(str(save_txt) + '\n')
    return "write \"" + str(save_txt) + "\" success!"

def writeLines2txt(save_path, lines):
    import os

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, mode="a", encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    return save_path


def split2train(path, rate=.8, type='shuffle'):
    import os

    import random
    with open(path, 'rt') as f:
        # dataSet = [ln.strip('\n').rstrip().split() for ln in f]
        # dataSet = [ln.strip('\n').rstrip().replace(' ', '_') for ln in f]
        dataSet = [ln.strip('\n').rstrip() for ln in f]

    pos_nums = round(len(dataSet) * rate)

    if type == 'shuffle':
        # 乱序
        random.shuffle(dataSet)
        # [训练集, 测试集]
        trainSet, testSet = dataSet[:pos_nums], dataSet[pos_nums:]

    elif type == 'sample':
        # 默认是不重复抽样（无放回抽样）
        trainSet = random.sample(dataSet, pos_nums)
        # testSet = [item for item in dataSet if item not in trainSet]
        testSet = list(set(dataSet) - set(trainSet))
    else:
        return "KO!"

    train_path = os.path.splitext(path)[0] + '_train.txt'
    for ln in trainSet:
        print(writetxt(train_path, ln))
    test_path = os.path.splitext(path)[0] + '_test.txt'
    for ln in testSet:
        print(writetxt(test_path, ln))

    return train_path, test_path


def increase2balance(path, type='shuffle'):
    import os

    import numpy
    from collections import Counter
    with open(path, 'rt') as f:
        alldata = [ln.strip('\n').rstrip().split() for ln in f]

    labels = [data[-1] for data in alldata]
    class_list = list(set(labels))
    class_count = dict(Counter(labels))
    class_count_max = max(class_count.values())
    dataSet = {}
    for label in class_list:
        dataSet[str(label)] = []
    for data in alldata:
        dataSet[str(data[-1])].append(data[0])

    for ds in dataSet:
        threshold = abs(len(dataSet[ds]) - class_count_max) / class_count_max
        if threshold > 0.1:
            # 默认是重复抽样（有放回抽样）
            dataSet[ds] = list(numpy.random.choice(dataSet[ds], class_count_max))

    save_path = os.path.splitext(path)[0] + '_balance.txt'

    # for ds in dataSet:
    #     for ln in dataSet[ds]:
    #         save_txt = ln + ' ' + ds
    #         print(writetxt(save_path, save_txt))

    save_lines = [ln + ' ' + ds for ds in dataSet for ln in dataSet[ds]]

    if type == 'shuffle':
        import random
        # 乱序
        random.shuffle(save_lines)
        writeLines2txt(save_path, save_lines)
    else:
        writeLines2txt(save_path, save_lines)

    return save_path


if __name__ == "__main__":
    txt_path = r'C:\xxx\xxx\cls\full\full.txt'
    # split2train(increase2balance(txt_path))

    train_txt, test_txt = split2train(txt_path, 0.9)
    increase2balance(train_txt)
    increase2balance(test_txt)
