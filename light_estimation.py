from utils.utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse
from tqdm import tqdm

from model.defineHourglass_512_gray_skip import HourglassNet


# ---------------- create normal for rendering half sphere ------
def create_normal(img_size=256):
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    return normal


def load_sh(fp="data/example_light/rotate_light_07.txt"):
    sh = np.loadtxt(fp)
    sh = sh[0:9]
    sh = sh * 0.7
    return sh.squeeze()


def render_half_sphere(normal, sh, img_size=256):

    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (img_size, img_size))

    valid = normal.reshape((img_size, img_size, 3))[..., 1] < 0
    shading = shading * valid

    return shading


# -----------------------------------------------------------------


def load_model(modelFolder="trained_model", **net_kwds):

    my_network = HourglassNet(**net_kwds)
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, "trained_model_03.t7")))
    my_network.cuda()
    my_network.train(False)

    return my_network


def process_img(img, sh, network):
    row, col, _ = img.shape
    img = cv2.resize(img, (512, 512))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, outputSH = network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    result_img = cv2.resize(resultLab, (col, row))

    est_sh = np.asarray(np.squeeze(outputSH).tolist())

    return result_img, est_sh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", default="images", type=str)
    parser.add_argument("dst_dir", default="results", type=str)
    parser.add_argument("--compare", "-c", action="store_true")
    args = parser.parse_args()

    target_sh = load_sh()
    normal = create_normal()
    net = load_model()
    target_sphere = render_half_sphere(normal, target_sh)
    target_sphere = cv2.cvtColor(target_sphere, cv2.COLOR_GRAY2BGR)

    paths = ["output_imgs", "light_spheres", "estimated_sh"]
    if args.compare:
        paths.append("comparison")

    paths = [os.path.join(args.dst_dir, folder) for folder in paths]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    for img_fn in tqdm(os.listdir(args.src_dir)):
        img = cv2.imread(os.path.join(args.src_dir, img_fn))
        output_img, est_sh = process_img(img, target_sh, net)
        light_sphere = render_half_sphere(normal, est_sh)

        img_name = os.path.splitext(img_fn)[0]
        cv2.imwrite(os.path.join(paths[0], img_fn), output_img)
        cv2.imwrite(os.path.join(paths[1], img_fn), light_sphere)
        np.savetxt(os.path.join(paths[2], img_name + ".txt"), est_sh, fmt="%.7f")
        if args.compare:
            rows = img.shape[0]
            sphere = cv2.resize(light_sphere, (rows, rows))
            bgr_sphere = cv2.cvtColor(sphere, cv2.COLOR_GRAY2BGR)
            comp = np.hstack((img, bgr_sphere, output_img, cv2.resize(target_sphere, (rows, rows))))
            cv2.imwrite(os.path.join(paths[3], img_fn), comp)


if __name__ == "__main__":
    main()
