# save_and_load_h5py.py
import os
import numpy as np
import cv2
import h5py


# 加载数据集中的文件
def save_image_to_h5py(path):
    img_list = []
    label_list = []
    dir_counter = 0
    # num_for_test = 0

    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        # print('文件中的子文件名是:\n', child_path)
        # 总共有9个文件夹 第一个文件夹加载10文件 其他文件夹中加载1个文件
        for dir_image in os.listdir(child_path):
            # print('dir_image中图像的名称是:\n', dir_image)
            img = cv2.imread(os.path.join(child_path, dir_image))
            # img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#单通道，分辨率会下降
            img_list.append(img)
            label_list.append(dir_counter)

            # if num_for_test > 10:
            #     break
            # num_for_test = num_for_test + 1

        # 返回的img_list转成了 np.array的格式
        dir_counter += 1

    img_np = np.array(img_list)
    label_np = np.array(label_list)
    print('数据集中原始的标签顺序是:\n', label_np)

    f = h5py.File('Dataset_GASF_train.h5', 'w')
    f['image'] = img_np
    f['labels'] = label_np
    f.close()


save_image_to_h5py('GASF20210315')


# 加载hdpy成np的形式
def load_h5py_to_np(path):
    h5_file = h5py.File(path, 'r')
    print('打印一下h5py中有哪些关键字', h5_file.keys())
    permutation = np.random.permutation(len(h5_file['labels']))
    shuffled_image = h5_file['image'][:][permutation, :, :, :]
    shuffled_label = h5_file['labels'][:][permutation]
    print('经过打乱之后数据集中的标签顺序是:\n', shuffled_label, len(h5_file['labels']))
    return shuffled_image, shuffled_label


images, labels = load_h5py_to_np('Dataset_GASF_train.h5')

# print(images[:1960])
train_set_y_orig = labels[:1960]
train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
print(type(images))
print(train_set_y_orig)
# print(labels)
# 读取并保存
# for i, image in enumerate(images):
#     cv2.imwrite("filename.png", image)
