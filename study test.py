import os
import cv2
import numpy as np
from UNet_Fusion_Net import *
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def RGBload_test():
    # src_Pan = "C:/Users/zpf/Desktop/SSdataset/test/PAN/"
    # src_Ms = "C:/Users/zpf/Desktop/SSdataset/test/MS/"

    src_Pan = "C:/Users/Administrator.SC-202008062009/Desktop/ssdata/test/PAN/PAN/"
    src_Ms = "C:/Users/Administrator.SC-202008062009/Desktop/ssdata/test/MS/MS/"

    # 加载数据
    x_train_list = os.listdir(src_Pan)
    y_test_list = os.listdir(src_Ms)

    Pan = np.empty((len(x_train_list), 3, 256, 256))  # 生成空的数组
    Ms = np.empty((len(y_test_list), 3, 64, 64))  # 生成空的数组

    i = 0
    j = 0

    for name in x_train_list:  # Pan仅导入灰度图
        temp = os.path.join(src_Pan, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        Pan[i][:, :, :] = img
        i = i + 1
    for name in y_test_list:  # Ms三个分量分开导入并融合
        temp = os.path.join(src_Ms, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        Ms[j][:, :, :] = img
        j = j + 1

    # flat
    Pan = np.array(Pan).astype('float32') / 255.
    Ms = np.array(Ms).astype('float32') / 255.

    Pan = torch.tensor(Pan)
    Pan = Variable(Pan, requires_grad=True)
    Ms = torch.tensor(Ms)
    Ms = Variable(Ms, requires_grad=True)

    return Pan, Ms, len(x_train_list)


def test():
    data = RGBload_test()

    model_test = model_rgb()
    model_test.cuda()

    model_test.load_state_dict(torch.load('./UNet_weights/best_UNet_436.pkl'))

    for i in range(data[2]):
        Pan = data[0][i:i + 1, :, :, :].cuda()
        Ms = data[1][i:i + 1, :, :, :].cuda()

        G = model_test(Pan, Ms)
        test_img_sav = img_recon_rgb(G)
        if not os.path.isdir('./testimg/'):
            os.makedirs('./testimg/')
        cv2.imwrite("./testimg/{}.png".format(i + 1), test_img_sav)

    print('ok')


test()
