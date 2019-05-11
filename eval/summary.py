import thop
from models.resnet import resnet34_cifar100, resnet50_cifar100
from models.senet import se_resnet34_cifar100, se_resnet50_cifar100
from models.bam import bam_resnet34_cifar100, bam_resnet50_cifar100
from models.cbam import cbam_resnet34_cifar100, cbam_resnet50_cifar100


models34 = [resnet34_cifar100(), se_resnet34_cifar100(), bam_resnet34_cifar100(), cbam_resnet34_cifar100()]
models50 = [resnet50_cifar100(), se_resnet50_cifar100(), bam_resnet50_cifar100(), cbam_resnet50_cifar100()]

flop_lst = list()
param_lst = list()

for model in (models34 + models50):
    model = model.cuda()
    flops, params = thop.profile(model=model, input_size=(1, 3, 32, 32), device='cuda')
    flop_lst.append(flops)
    param_lst.append(params)
else:
    print('\n\nFlops are ', end='')
    print(*flop_lst)
    print('Param numbers are ', end='')
    print(*param_lst)


