import os
import cv2
import torch
from torch.autograd import Variable
import numpy as np
from model.defineHourglass_512_gray_skip import HourglassNet


def load_model(modelFolder="trained_model", **net_kwds):

    my_network = HourglassNet(**net_kwds)
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, "trained_model_03.t7")))
    my_network.cuda()
    my_network.train(False)

    return my_network


def image_generator(images_fn, directory="./", batch_size=32, target_size=(512, 512)):

    batch_imgs = []
    for img_fn in images_fn:
        img = cv2.imread(os.path.join(directory, img_fn))
        img = cv2.resize(img, target_size)
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:, :, 0]
        inputL = inputL.astype(np.float32) / 255.0
        inputL = inputL.transpose((0, 1))

        batch_imgs.append(inputL[None, ...])

        if len(batch_imgs) == batch_size:
            batch = np.stack(batch_imgs)
            batch = Variable(torch.from_numpy(batch).cuda())
            yield batch

            batch_imgs = []
                    
    if batch_imgs:
        batch = np.stack(batch_imgs)
        batch = Variable(torch.from_numpy(batch).cuda())
        yield batch
