from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip


def train_transform():
    return Compose([RandomHorizontalFlip(), ToTensor()])


def val_transform():
    return ToTensor()
