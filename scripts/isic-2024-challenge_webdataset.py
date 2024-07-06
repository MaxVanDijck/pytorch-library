import polars as pl
import json
import webdataset as wds

if __name__ == "__main__":
    sink = wds.TarWriter("../data/isic-2024-challenge/webdataset.tar")

    df = pl.read_csv("../data/isic-2024-challenge/train-metadata.csv")
    for row in df.iter_rows(named=True):
        with open(f"../data/isic-2024-challenge/train-image/image/{row['isic_id']}.jpg", "rb") as stream:
            image = stream.read()

        sink.write({
                "__key__": row['isic_id'],
                "metadata.json": json.dumps(row),
                "image.jpg": image,
            })

    extra_df = pl.read_csv("../data/isic-2024-challenge/extras/metadata.csv")
    for row in extra_df.iter_rows(named=True):
        with open(f"../data/isic-2024-challenge/extras/{row['isic_id']}.jpg", "rb") as stream:
            image = stream.read()

        sink.write({
                "__key__": row['isic_id'],
                "metadata.json": json.dumps(row),
                "image.jpg": image,
            })
