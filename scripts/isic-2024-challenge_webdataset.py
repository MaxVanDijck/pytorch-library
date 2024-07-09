import polars as pl
import json
import webdataset as wds
from PIL import Image
import io

def determine_label(metadata: dict) -> int | None:
    try:
        if metadata["iddx_1"] == "Benign":
            return 0

        if metadata["iddx_1"] == "Indeterminate":
            return 1

        if metadata["iddx_1"] == "Malignant":
            return 1

    except Exception:
        pass

    try:
        if metadata["benign_malignant"] == "benign":
            return 0

        if metadata["benign_malignant"] == "indeterminate":
            return 1

        if metadata["benign_malignant"] == "indeterminate/benign":
            return 0

        if metadata["benign_malignant"] == "indeterminate/malignant":
            return 1

        if metadata["benign_malignant"] == "malignant":
            return 1
    except Exception:
        pass

    return None

def resize_and_crop_image(image_bytes, target_size=(128, 128)):
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Determine the aspect ratio
    aspect_ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
    
    # Resize the image, keeping the aspect ratio
    new_size = (int(image.size[0] * aspect_ratio), int(image.size[1] * aspect_ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    # Calculate cropping coordinates to center the image
    left = (image.size[0] - target_size[0]) / 2
    top = (image.size[1] - target_size[1]) / 2
    right = (image.size[0] + target_size[0]) / 2
    bottom = (image.size[1] + target_size[1]) / 2
    
    # Crop the image to the target size
    image = image.crop((left, top, right, bottom))
    
    # Save the resulting image to a bytes object
    output_bytes = io.BytesIO()
    image.save(output_bytes, format='JPEG')
    
    return output_bytes.getvalue()

if __name__ == "__main__":
    positives = 0
    negatives = 0
    positive_file_num = 0
    negative_file_num = 0
    negative_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-negative-2024_{str(negative_file_num).zfill(6)}.tar")
    positive_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-positive-2024_{str(positive_file_num).zfill(6)}.tar")

    df = pl.read_csv("../data/isic-2024-challenge/train-metadata.csv")
    for row in df.iter_rows(named=True):
        with open(f"../data/isic-2024-challenge/train-image/image/{row['isic_id']}.jpg", "rb") as stream:
            image = resize_and_crop_image(stream.read())

        label = determine_label(row)
        if label == 1:
            positive_sink.write({
                    "__key__": row['isic_id'],
                    "metadata.json": json.dumps(row),
                    "image.jpg": image,
                })

            positives += 1
            if positives % 100 == 0:
                positive_file_num += 1
                positive_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-positive-2024_{str(positive_file_num).zfill(6)}.tar")

        elif label == 0:
            negative_sink.write({
                    "__key__": row['isic_id'],
                    "metadata.json": json.dumps(row),
                    "image.jpg": image,
                })

            negatives += 1
            if negatives % 100 == 0:
                negative_file_num += 1
                negative_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-negative-2024_{str(negative_file_num).zfill(6)}.tar")



    positives = 0
    negatives = 0
    positive_file_num = 0
    negative_file_num = 0
    negative_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-negative-old_{str(negative_file_num).zfill(6)}.tar")
    positive_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-positive-old_{str(positive_file_num).zfill(6)}.tar")
    extra_df = pl.read_csv("../data/isic-2024-challenge/extras/metadata.csv")
    for row in extra_df.iter_rows(named=True):
        with open(f"../data/isic-2024-challenge/extras/{row['isic_id']}.jpg", "rb") as stream:
            image = resize_and_crop_image(stream.read())

        label = determine_label(row)
        if label == 1:
            positive_sink.write({
                    "__key__": row['isic_id'],
                    "metadata.json": json.dumps(row),
                    "image.jpg": image,
                })

            positives += 1
            if positives % 100 == 0:
                positive_file_num += 1
                positive_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-positive-old_{str(positive_file_num).zfill(6)}.tar")

        elif label == 0:
            negative_sink.write({
                    "__key__": row['isic_id'],
                    "metadata.json": json.dumps(row),
                    "image.jpg": image,
                })

            negatives += 1
            if negatives % 100 == 0:
                negative_file_num += 1
                negative_sink = wds.TarWriter(f"../data/isic-2024-challenge/webdataset-negative-old_{str(negative_file_num).zfill(6)}.tar")
