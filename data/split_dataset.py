import os
from os import path


def scaner_file(url):
    files = os.listdir(url)
    for f in files:
        real_url = path.join(url, f)
        if path.isfile(real_url):
            # print(path.abspath(real_url))
            # 如果是文件，则以绝度路径的方式输出

            with open(path.abspath(real_url), mode='r', encoding='utf-8') as file:
                line = file.readlines()  # 读取文件
                print(len(line))
                # try:
                #     line = line[1:]  # 只读取第一行之后的内容
                #     file = open(path.abspath(real_url), mode='w', encoding='utf-8')  # 以写入的形式打开txt文件
                #     file.writelines(line)  # 将修改后的文本内容写入
                #     file.close()  # 关闭文件
                # except:
                #     pass

        elif path.isdir(real_url):
            # 如果是目录，则是递归调用自定义函数 scaner_file (url)进行多次
            scaner_file(real_url)
        else:

            # print("其他情况")
            pass
        # print(real_url)


# txt or other splits in the file to remove unimportant information
scaner_file("./single_cell/")

