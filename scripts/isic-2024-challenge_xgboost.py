import ray
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
import polars as pl

if __name__ == "__main__":
    ray.init()
    df = pl.read_csv("../data/isic-2024-challenge/train-metadata.csv")
    df = df.drop([
        "isic_id",
        "sex", # potentially useful
        "anatom_site_general",
        "image_type",
        "tbp_tile_type",
        "attribution",
        "copyright_license",
        "lesion_id",

        # LABELS
        "iddx_full",
        "iddx_1",
        "iddx_2",
        "iddx_3",
        "iddx_4",
        "iddx_5",
        "mel_thick_mm", 
        "mel_mitotic_index", 
        "tbp_lv_dnn_lesion_confidence",

        # Strings
        "tbp_lv_location",
        "tbp_lv_location_simple",
        "age_approx",
    ])

    patient_ids = df["patient_id"].unique().to_list()

    test_patients = patient_ids[0:int(len(patient_ids) // 5)]
    train_patients = patient_ids[int(len(patient_ids) // 5):]

    train_df = df.filter(pl.col('patient_id').is_in(train_patients)).drop("patient_id")
    test_df = df.filter(pl.col('patient_id').is_in(test_patients)).drop("patient_id")

    train_df.write_parquet("../data/isic-2024-challenge/train_tabular.parquet")
    test_df.write_parquet("../data/isic-2024-challenge/test_tabular.parquet")

    # Train xgboost ray
    train_ds = ray.data.read_parquet("../data/isic-2024-challenge/train_tabular.parquet")
    test_ds = ray.data.read_parquet("../data/isic-2024-challenge/test_tabular.parquet")



    trainer = XGBoostTrainer(
        label_column="target",
        num_boost_round=20,
        params={
            # XGBoost specific params (see the `xgboost.train` API reference)
            "objective": "binary:logistic",
            # uncomment this and set `use_gpu=True` to use GPU for training
            # "tree_method": "gpu_hist",
            "eval_metric": ["logloss", "error"],
        },
        datasets={"train": train_ds, "valid": test_ds},
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )
    result = trainer.fit()
    print(result.metrics)
    breakpoint()
