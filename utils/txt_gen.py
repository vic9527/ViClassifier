def walkdir(dir):
    import os

    flist = []
    for root, dirs, files in os.walk(dir):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # # 遍历文件
        # for f in files:
        #     print(os.path.join(root, f))

        # flist.append([os.path.join(root, f) for f in files])
        for f in files:
            flist.append(os.path.join(root, f))

        # # 遍历所有的文件夹
        # for d in dirs:
        #     print(os.path.join(root, d))
    return flist


def writetxt(save_path, save_txt):
    import os

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, mode="a", encoding='utf-8') as f:
        f.write(str(save_txt) + '\n')
    return "write \"" + str(save_txt) + "\" success!"

def gentxt(image_dir):
    import os

    ext = ('.jpg', 'jpeg', 'png')
    for image_path in walkdir(image_dir):
        # print(image_path)
        if image_path.strip().find(' ') >= 0:  # 判断路径是否有空格
            os.rename(image_path, image_path.replace(' ', '_'))
            image_path = image_path.replace(' ', '_')
        if image_path.endswith(ext):
            image_folder = os.path.dirname(image_path)
            image_name = os.path.basename(image_path)
            # print(image_name, os.path.split(image_folder)[-1])
            save_path = os.path.join(os.path.split(image_folder)[0], "full.txt")
            save_txt = image_name + ' ' + os.path.split(image_folder)[-1]
            print(writetxt(save_path, save_txt))
    return "Done!"



if __name__ == '__main__':
    image_dir = r'C:\xxx\xxx\cls\full'

    # for case in os.listdir(image_dir):
    #     dir = os.path.join(image_dir, case)
    #     if not os.path.isdir(dir):
    #         continue
    #     print(gentxt(dir))

    print(gentxt(image_dir))