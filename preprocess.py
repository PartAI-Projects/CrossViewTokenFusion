import ast
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pydicom import dcmread
from skimage.io import imsave


def crop_breast_region(image, min_pixel=10):
    _, binary = cv2.threshold(
        image,
        min_pixel,
        255,
        cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    cropped = image[y:y + h, x:x + w]
    return cropped

def apply_clahe(image, clip_limit):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )
    return clahe.apply(image)

def preprocess_single_image(
        image,
        clip_limit=1.0,
        min_pixel=10,
        target_size=(448, 448)
):
    """
    Crop -> CLAHE -> Resize
    """
    # Ensure uint8 before OpenCV operations
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    image = crop_breast_region(image, min_pixel)
    image = apply_clahe(image, clip_limit)
    image = cv2.resize(
        image,
        target_size,
        interpolation=cv2.INTER_AREA
    )
    return image

def preprocess_cmmd(
        data_dir,
        output_dir,
        image_size,
        clip_limit
):
    """
    DICOM -> flip -> crop -> CLAHE -> resize -> save PNG
    """
    root_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dicom_paths = list(root_dir.rglob("*.dcm"))
    print(f"Found {len(dicom_paths)} DICOM files")

    for dicom_path in tqdm(dicom_paths, desc="Processing CMMD"):

        dicom = dcmread(str(dicom_path))
        image = dicom.pixel_array

        # Flip right breast
        laterality = dicom.get((0x0020, 0x0062), None)
        if laterality is not None:
            laterality = laterality.value
        if laterality == "R":
            image = image[:, ::-1]

        # Preprocessing
        image = preprocess_single_image(
            image=image,
            clip_limit=clip_limit,
            target_size=image_size
        )
        # Preserve original folder structure
        rel_path = dicom_path.relative_to(root_dir)
        save_path = (
            output_dir /
            rel_path.parent /
            f"{dicom_path.stem}.png"
        )
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        cv2.imwrite(str(save_path), image)

def preprocess_vindr(
        data_dir,
        output_dir,
        image_size,
        clip_limit
):
    """
    DICOM -> flip -> crop -> CLAHE -> resize -> PNG
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        data_dir / 'breast-level_annotations.csv'
    )
    meta = pd.read_csv(
        data_dir / 'metadata.csv'
    )
    meta = meta.rename(
        columns={'SOP Instance UID': 'image_id'}
    )
    data = data.merge(meta, on='image_id')

    for idx in tqdm(range(len(data)), desc="Processing VinDr"):
        study_id = data.loc[idx, 'study_id']
        image_id = data.loc[idx, 'image_id']
        dicom_path = (
            data_dir /
            'images' /
            str(study_id) /
            f'{image_id}.dicom'
        )

        save_dir = output_dir / str(study_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{image_id}.png'

        dicom = dcmread(str(dicom_path))
        image = dicom.pixel_array

        window = np.array(
            ast.literal_eval(
                data.loc[idx, 'Window Width']
            )
        )
        level = np.array(
            ast.literal_eval(
                data.loc[idx, 'Window Center']
            )
        )

        # GIOTTO machines provide multiple values
        if data.loc[idx,
                    "Manufacturer's Model Name"] in [
                        'GIOTTO IMAGE 3DL',
                        'GIOTTO CLASS'
                    ]:

            window = window[0]
            level = level[0]

        # Handle MONOCHROME1 images
        if data.loc[idx,
                    'Photometric Interpretation'] == 'MONOCHROME1':

            image[image == 1] += data.loc[idx, 'Pixel Padding Value']
            level = np.max(image) - level
            image = np.max(image) - image

        image = image.astype(np.float32)
        image -= (level - window / 2)
        image /= window
        image[image < 0] = 0
        image[image > 1] = 1
        image *= 255

        if data.loc[idx, 'laterality'] == 'R':
            image = image[:, ::-1]

        image = preprocess_single_image(
            image=image,
            clip_limit=clip_limit,
            target_size=image_size
        )
        imsave(str(save_path), image)


def preprocess(cfg):
    data_name = cfg.data.data_name.lower()
    if data_name == "vindr":
        preprocess_vindr(
            data_dir=cfg.data.data_dir,
            output_dir=cfg.data.prep_dir,
            image_size=cfg.data.image_size,
            clip_limit=2.0
        )
    elif data_name == "cmmd":
        preprocess_cmmd(
            data_dir=cfg.data.data_dir,
            output_dir=cfg.data.prep_dir,
            image_size=cfg.data.image_size,
            clip_limit=1.0
        )
    else:
        raise ValueError(f"Unknown dataset: {data_name}")