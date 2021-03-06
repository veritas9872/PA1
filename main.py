from train.training import train_model

from utils.run_utils import create_arg_parser

from models.senet import se_resnet34_cifar100, se_resnet50_cifar100
from models.bam import bam_resnet34_cifar100, bam_resnet50_cifar100
from models.cbam import cbam_resnet34_cifar100, cbam_resnet50_cifar100

from torchvision.models import resnet34, resnet50

from models.my_module import my_resnet34_cifar100, my_resnet50_cifar100


def models34(args):
    resnet = resnet34(num_classes=100)

    senet = se_resnet34_cifar100()

    bam_ca_sa = bam_resnet34_cifar100(use_ca=True, use_sa=True)
    bam_ca = bam_resnet34_cifar100(use_ca=True, use_sa=False)  # Not the same as SENet!
    bam_sa = bam_resnet34_cifar100(use_ca=False, use_sa=True)
    # bam_no = bam_resnet34_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    cbam_ca_sa = cbam_resnet34_cifar100(use_ca=True, use_sa=True)
    cbam_ca = cbam_resnet34_cifar100(use_ca=True, use_sa=False)  # Not the same as SENet!
    cbam_sa = cbam_resnet34_cifar100(use_ca=False, use_sa=True)
    # cbam_no = cbam_resnet34_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    train_model(resnet, args=args)  # 1
    train_model(senet, args=args)  # 2
    train_model(bam_ca_sa, args=args)  # 3
    train_model(bam_ca, args=args)  # 4
    train_model(bam_sa, args=args)  # 5
    train_model(cbam_ca_sa, args=args)  # 6
    train_model(cbam_ca, args=args)  # 7
    train_model(cbam_sa, args=args)  # 8


def models50(args):
    resnet = resnet50(num_classes=100)

    senet = se_resnet50_cifar100()

    bam_ca_sa = bam_resnet50_cifar100(use_ca=True, use_sa=True)
    bam_ca = bam_resnet50_cifar100(use_ca=True, use_sa=False)  # Not the same as SENet!
    bam_sa = bam_resnet50_cifar100(use_ca=False, use_sa=True)
    # bam_no = bam_resnet50_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    cbam_ca_sa = cbam_resnet50_cifar100(use_ca=True, use_sa=True)
    cbam_ca = cbam_resnet50_cifar100(use_ca=True, use_sa=False)  # Not the same as SENet!
    cbam_sa = cbam_resnet50_cifar100(use_ca=False, use_sa=True)
    # cbam_no = cbam_resnet50_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    train_model(resnet, args=args)  # 1
    train_model(senet, args=args)  # 2
    train_model(bam_ca_sa, args=args)  # 3
    train_model(bam_ca, args=args)  # 4
    train_model(bam_sa, args=args)  # 5
    train_model(cbam_ca_sa, args=args)  # 6
    train_model(cbam_ca, args=args)  # 7
    train_model(cbam_sa, args=args)  # 8


def my_model(args):
    # Setting pool_stride=1 disables it. Necessary because the CIFAR 100 images are so small.
    my_model34 = my_resnet34_cifar100(reduction_ratio=16, dilation_value=2, pool_stride=1, use_ca=True, use_sa=True)
    my_model50 = my_resnet50_cifar100(reduction_ratio=16, dilation_value=2, pool_stride=1, use_ca=True, use_sa=True)

    train_model(my_model34, args=args)
    train_model(my_model50, args=args)


if __name__ == '__main__':
    defaults = dict(
        batch_size=12,
        num_workers=1,
        init_lr=0.001,
        gamma=0.1,  # Factor by which to reduce lr.
        step_size=20,
        gpu=0,  # Set to None for CPU mode.
        num_epochs=30,
        verbose=False,
        save_best_only=True,
        max_to_keep=1,
        data_root='/home/veritas/PycharmProjects/PA1/data',
        ckpt_root='/home/veritas/PycharmProjects/PA1/checkpoints',
        log_root='/home/veritas/PycharmProjects/PA1/logs'
    )

    parser = create_arg_parser(**defaults).parse_args()

    # models34(parser)
    # models50(parser)
    my_model(parser)
