from train.training import train_model

from utils.run_utils import create_arg_parser

from models.senet import se_resnet34_cifar100, se_resnet50_cifar100
from models.bam import bam_resnet34_cifar100, bam_resnet50_cifar100
from models.cbam import cbam_resnet34_cifar100, cbam_resnet50_cifar100

from torchvision.models import resnet34, resnet50


def models34(args):
    resnet = resnet34(num_classes=100)

    senet = se_resnet34_cifar100()

    bam_ca_sa = bam_resnet34_cifar100(use_ca=True, use_sa=True)
    # bam_ca = bam_resnet34_cifar100(use_ca=True, use_sa=False)  # Same as SENet
    bam_sa = bam_resnet34_cifar100(use_ca=False, use_sa=True)
    # bam_no = bam_resnet34_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    cbam_ca_sa = cbam_resnet34_cifar100(use_ca=True, use_sa=True)
    # cbam_ca = cbam_resnet34_cifar100(use_ca=True, use_sa=False)  # Same as SENet
    cbam_sa = cbam_resnet34_cifar100(use_ca=False, use_sa=True)
    # cbam_no = cbam_resnet34_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    train_model(resnet, args=args)  # 1
    train_model(senet, args=args)  # 2
    train_model(bam_ca_sa, args=args)  # 3
    train_model(bam_sa, args=args)  # 4
    train_model(cbam_ca_sa, args=args)  # 5
    train_model(cbam_sa, args=args)  # 6


def models50(args):
    resnet = resnet50(num_classes=100)

    senet = se_resnet50_cifar100()

    bam_ca_sa = bam_resnet50_cifar100(use_ca=True, use_sa=True)
    # bam_ca = bam_resnet50_cifar100(use_ca=True, use_sa=False)  # Same as SENet
    bam_sa = bam_resnet50_cifar100(use_ca=False, use_sa=True)
    # bam_no = bam_resnet50_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    cbam_ca_sa = cbam_resnet50_cifar100(use_ca=True, use_sa=True)
    # cbam_ca = cbam_resnet50_cifar100(use_ca=True, use_sa=False)  # Same as SENet
    cbam_sa = cbam_resnet50_cifar100(use_ca=False, use_sa=True)
    # cbam_no = cbam_resnet50_cifar100(use_ca=False, use_sa=False)  # Same as ResNet

    train_model(resnet, args=args)  # 1
    train_model(senet, args=args)  # 2
    train_model(bam_ca_sa, args=args)  # 3
    train_model(bam_sa, args=args)  # 4
    train_model(cbam_ca_sa, args=args)  # 5
    train_model(cbam_sa, args=args)  # 6


if __name__ == '__main__':
    defaults = dict(
        batch_size=12,
        num_workers=1,
        init_lr=0.001,
        gamma=0.1,  # Factor by which to reduce lr.
        step_size=25,
        gpu=0,  # Set to None for CPU mode.
        num_epochs=50,
        verbose=False,
        save_best_only=True,
        max_to_keep=5,
        data_root='/home/veritas/PycharmProjects/PA1/data',
        ckpt_root='/home/veritas/PycharmProjects/PA1/checkpoints',
        log_root='/home/veritas/PycharmProjects/PA1/logs'
    )

    parser = create_arg_parser(**defaults).parse_args()

    # models34(parser)
    models50(parser)

