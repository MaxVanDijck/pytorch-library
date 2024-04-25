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
        "train": get_ray_dataset('train'), 
        "valid": get_ray_dataset('validation'),
    }

def get_dataset_collate_fns():
    return {
        "train": collate_fn_train,
        "valid": collate_fn_valid,
    }


def get_ray_dataset(split):
    hf_dataset = datasets.load_dataset("frgfm/imagenette", '320px')
    ray_ds = ray.data.from_huggingface(hf_dataset[split]).random_shuffle()
    return ray_ds


def collate_fn_train(batch):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize(128)
    crop = transforms.CenterCrop(128)
    erasing = transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    x = []
    for item in batch['image']:
        image = pil_to_tensor(Image.open(io.BytesIO(item['bytes'])))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = resize(image)

        image = crop(image).to(torch.float32)
        image = normalize(image)
        image = erasing(image)
        x.append(image)
    x = torch.stack(x).to(torch.float32)
    y = torch.tensor(batch['label'], dtype=torch.uint8)
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
    y = torch.tensor(batch['label'], dtype=torch.uint8)
    return x, y

