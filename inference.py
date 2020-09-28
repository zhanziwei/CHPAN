#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from pspnet import PSPNet

models = {
    'squeezenet': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet',pretrained=False),
    'densenet': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet',pretrained=False),
    'resnet18': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18',pretrained=False),
    'resnet34': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',pretrained=False),
    'resnet50': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',pretrained=False),
    'resnet101': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101',pretrained=False),
    'resnet152': lambda: PSPNet(n_classes=20,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152',pretrained=False)
}

parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
parser.add_argument('--image_path', default='E:\workspace_python\HPA\Single-Human-Parsing-LIP-master\demo/test2.png',type=str, help='Path to image')
parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='resnet101', help='Feature extractor')
parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
args = parser.parse_args()


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    # net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        # model = torch.load(model_path, map_location='cuda:0')
        net.load_state_dict(torch.load(snapshot,map_location='cuda:0'))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)


def show_image(img, pred):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img.cpu().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))

    # show image
    ax0.set_title('img')
    # img = img.resize((128, 386), Image.BILINEAR)
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=0.7, )
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(2.3, (j + 0.45) / 20.0, lab, ha='left', va='center', )

    plt.savefig(fname="./result.jpg")
    print('result saved to ./result.jpg')
    plt.show()


def main():
    # --------------- model --------------- #
    # str_ids = args.gpu_ids.split(',')
    # gpu_ids = []
    # for str_id in str_ids:
    #     gid = int(str_id)
    #     if gid >= 0:
    #         gpu_ids.append(gid)
    # # set gpu ids
    # if len(gpu_ids) > 0:
    #     torch.cuda.set_device(gpu_ids[0])

    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    # ------------ load image ------------ #
    data_transform = get_transform()
    img = Image.open(args.image_path)
    img = data_transform(img)
    img = img.cuda()

    # --------------- inference --------------- #

    with torch.no_grad():
        pred, _ = net(img.unsqueeze(dim=0))
        pred = pred.squeeze(dim=0)
        pred = pred.cpu().numpy().transpose(1, 2, 0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))

        show_image(img, pred)


if __name__ == '__main__':
    main()