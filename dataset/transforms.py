import albumentations as A


def get_transforms():
    
    return A.Compose([
        A.Rotate(
            limit=3,
            interpolation=1,
            border_mode=0,
            p=0.3
        ),

        A.Affine(
            translate_percent=0.05,
            scale=(0.95, 1.05),
            p=0.2
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.2
        ),

        A.RandomGamma(
            gamma_limit=(90, 110),
            p=0.1
        ),

        A.GaussNoise(
            std_range=(0.0, 0.01),
            p=0.1
        ),
    ])
