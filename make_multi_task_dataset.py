import numpy as np
import pandas as pd
from tqdm import tqdm

from bin.checkpoints.upload_to_kaggle import (
    kaggle_get_metadata,
    kaggle_new_dataset_version,
)
from bin.file_utils.rm_new_folder import rm_and_new_folder


def make_dataset():
    train = pd.read_csv("data/nbme-score-clinical-patient-notes/train.csv")
    features = pd.read_csv("data/nbme-score-clinical-patient-notes/features.csv")
    patient_notes = pd.read_csv(
        "data/nbme-score-clinical-patient-notes/patient_notes.csv"
    )
    # get_new_entities(patient_notes)
    train_merge_features = pd.merge(
        train,
        features,
        how="left",
        left_on=["case_num", "feature_num"],
        right_on=["case_num", "feature_num"],
    )

    train_merge_patient_notes = pd.merge(
        train_merge_features,
        patient_notes,
        how="left",
        left_on=["pn_num", "case_num"],
        right_on=["pn_num", "case_num"],
    )
    dataset_path = "data/nbme-merge-dataset"
    rm_and_new_folder(dataset_path)
    train_merge_patient_notes.to_csv(dataset_path + "/train.csv")
    kaggle_get_metadata(dataset_path, "nbme-merge-dataset")
    kaggle_new_dataset_version(dataset_path)


if __name__ == "__main__":
    make_dataset()
