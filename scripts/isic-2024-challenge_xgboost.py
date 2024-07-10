import ray
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
import polars as pl
import pandas as pd
import xgboost as xgb



if __name__ == "__main__":
    df = pl.read_csv("../data/isic-2024-challenge/train-metadata.csv")
    cols = [
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

    ]
    df = df.drop(cols)

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
        scaling_config=ScalingConfig(
            # Number of workers to use for data parallelism.
            num_workers=1,
            # Whether to use GPU acceleration. Set to True to schedule GPU workers.
            use_gpu=False,
        ),
        label_column="target",
        num_boost_round=50,
        params={
            # XGBoost specific params (see the `xgboost.train` API reference)
            "objective": "binary:logistic",
            # uncomment this and set `use_gpu=True` to use GPU for training
            # "tree_method": "gpu_hist",
            "eval_metric": ["auc"],
        },
        datasets={"train": train_ds, "valid": test_ds},
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )
    result = trainer.fit()
    print(result.metrics)
    
    checkpoint = result.checkpoint

    class Predictor:
        def __init__(self):
            self.model = XGBoostTrainer.get_model(checkpoint)

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            orig_df = df
            if df.empty:
                return df

            df = df.drop(["isic_id"], axis=1)

            x = xgb.DMatrix(df)
            orig_df["target"] = self.model.predict(x).tolist()
            orig_df = orig_df.drop(list(set(orig_df.columns) - set(["target", "isic_id"])), axis=1)
            return orig_df

    test_csv = pd.read_csv("../data/isic-2024-challenge/test-metadata.csv")
    test_csv = test_csv.drop([col for col in list(set(test_csv.columns) - set(train_df.columns) - set(["isic_id"])) + ["target"] if col in test_csv.columns], axis=1)

    inferenced_dataset = (ray.data.from_pandas(test_csv)
        .map_batches(Predictor, batch_size=2048, batch_format="pandas", concurrency=1)).to_pandas()

    breakpoint()

