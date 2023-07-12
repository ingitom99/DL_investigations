import torchvision.transforms as T

def get_augment():
    augment = T.Compose([T.RandomResizedCrop(28, scale=(0.1, 1.0), 
                     interpolation=T.InterpolationMode('bicubic')),
                     T.RandomHorizontalFlip(),
                     T.ToTensor(),
                     T.Normalize((0.5,), (0.5,))
                     ])
    return augment