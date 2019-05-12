import torch
import torch.nn as nn

from pathlib import Path

from eval.visualization_functions import save_class_activation_images

from eval.gradcam import GradCam

from PIL import Image

from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from models.resnet import resnet34_cifar100


def load_model(model, ckpt_dir):
    assert isinstance(model, nn.Module)
    save_dict = torch.load(ckpt_dir)
    model.load_state_dict(save_dict['model_state_dict'])


def make_grad_cam_imgs(model, name, data_root=None):
    assert isinstance(model, nn.Module)

    grad_cam = GradCam(model, target_layer='layer4', target_conv='conv2')

    data_root = '/home/veritas/PycharmProjects/PA1/data'

    val_dataset = CIFAR100(root=data_root, train=False)

    for idx, (image, target) in enumerate(val_dataset, start=1):
        print(target)
        tensor = (ToTensor()(image)).unsqueeze(dim=0)
        print(tensor.shape)
        cam = grad_cam.generate_cam(input_image=tensor, target_class=target)
        save_class_activation_images(org_img=image, activation_map=cam, file_name=f'{name}_{idx:02d}')

        if idx >= 10:
            break


def main():
    resnet = resnet34_cifar100()
    ckpt_dir = '/home/veritas/PycharmProjects/PA1/checkpoints/Trial 01  2019-05-10 00-00-27/ckpt_018.tar'
    load_model(resnet, ckpt_dir)
    make_grad_cam_imgs(resnet, name='resnet34')


if __name__ == '__main__':
    main()
