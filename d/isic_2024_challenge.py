import ray
import datasets
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch
from PIL import Image
import io

# TODO:
# document that the user needs to add these two functions with the correct keys and return types
def get_dataset(): 
    return {
        "train": get_ray_dataset(0, 80, 0, 3, 0, 0), 
        "valid": get_ray_dataset(1000, 1020, 3, 5, 0, 0), 
    }

def get_dataset_collate_fns():
    return {
        "train": collate_fn_train,
        "valid": collate_fn_train,
    }


def get_ray_dataset(start: int, end: int, pos_start: int, pos_end: int, pos_old_start: int, pos_old_end: int):
    return ray.data.read_webdataset(
            [f"~/max/pytorch-library/data/isic-2024-challenge/webdataset-negative-2024_{str(i).zfill(6)}.tar" for i in range(start, end)] +
            [f"~/max/pytorch-library/data/isic-2024-challenge/webdataset-positive-2024_{str(i).zfill(6)}.tar" for i in range(pos_start, pos_end)] +
            [f"~/max/pytorch-library/data/isic-2024-challenge/webdataset-positive-old_{str(i).zfill(6)}.tar" for i in range(pos_old_start, pos_old_end)],
            shuffle="files",
        ).randomize_block_order()


def collate_fn_train(batch):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize(128)
    crop = transforms.CenterCrop(128)
    vflip = transforms.RandomVerticalFlip(p=0.5)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    erasing = transforms.RandomErasing(p=0.4, scale=(0.02, 0.33))

    x = []
    labels = []
    for metadata, item in zip(batch['metadata.json'], batch['image.jpg']):
        image = torch.from_numpy(item.transpose((2,0,1)))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = resize(image)

        image = crop(image).to(torch.float32)
        image = vflip(hflip(image))
        image = image / 256
        image = normalize(image)
        image = erasing(image)

        try:
            if metadata["iddx_1"] == "Benign":
                labels.append(0)
                x.append(image)

            if metadata["iddx_1"] == "Indeterminate":
                labels.append(1)
                x.append(image)

            if metadata["iddx_1"] == "Malignant":
                labels.append(1)
                x.append(image)

        except Exception as e:
            pass

        try:
            if metadata["benign_malignant"] == "benign":
                labels.append(0)
                x.append(image)

            if metadata["benign_malignant"] == "indeterminate":
                labels.append(1)
                x.append(image)

            if metadata["benign_malignant"] == "indeterminate/benign":
                labels.append(0)
                x.append(image)

            if metadata["benign_malignant"] == "indeterminate/malignant":
                labels.append(1)
                x.append(image)

            if metadata["benign_malignant"] == "malignant":
                labels.append(1)
                x.append(image)
        except Exception as e:
            pass

    x = torch.stack(x).to(torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


def collate_fn_valid(batch):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize(128)
    crop = transforms.CenterCrop(128)
    x = []
    for item in batch['image']:
        image = pil_to_tensor(Image.open(io.BytesIO(item['bytes'])))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = resize(image)
        image = crop(image).to(torch.float32)
        image = normalize(image)
        x.append(image)
    x = torch.stack(x).to(torch.float32)
    y = torch.tensor(batch['label'], dtype=torch.float32)
    return x, y

