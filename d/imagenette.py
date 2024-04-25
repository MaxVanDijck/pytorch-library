import ray
import datasets

def get_ray_dataset(split):
    hf_dataset = datasets.load_dataset("frgfm/imagenette", '320px')
    ray_ds = ray.data.from_huggingface(hf_dataset[split]).random_shuffle()
    return ray_ds

def get_dataset(): 
    return {
        "train": get_ray_dataset('train'), 
        "valid": get_ray_dataset('validation'),
    }
