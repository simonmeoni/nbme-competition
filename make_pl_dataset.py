import numpy as np
import pandas as pd
from tqdm import tqdm

from bin.checkpoints.upload_to_kaggle import (
    kaggle_get_metadata,
    kaggle_new_dataset_version,
)
from bin.file_utils.rm_new_folder import rm_and_new_folder


def make_dataset():
    np.random.seed(42)
    train = pd.read_csv("data/nbme-score-clinical-patient-notes/train.csv")
    features = pd.read_csv("data/nbme-score-clinical-patient-notes/features.csv")
    patient_notes = pd.read_csv(
        "data/nbme-score-clinical-patient-notes/patient_notes.csv"
    )
    pl_patient_notes = patient_notes[
        ~patient_notes["pn_num"].isin(train["pn_num"].unique())
    ]
    np.random.choice(train["feature_num"].unique())
    id = []
    case_num = []
    pn_num = []
    feature_num = []
    feature_text = []
    for _, row in pl_patient_notes.iterrows():
        feature_to_choose = list(features[features['case_num'] == row['case_num']]['feature_num'].unique())
        for i in range(2):
            feature = np.random.choice(
                feature_to_choose
            )
            pn_num.append(row["pn_num"])
            case_num.append(row["case_num"])
            feature_num.append(feature)
            id.append("00" + str(row["pn_num"]) + "_00" + str(feature))
            feature_to_choose.remove(feature)
    train_pl = pd.DataFrame(
        data={
            "id": id,
            "case_num": case_num,
            "pn_num": pn_num,
            "feature_num": feature_num,
        }
    )
    train_merge_features = pd.merge(
        train_pl,
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
    dataset_path = "data/nbme-pl-dataset"
    rm_and_new_folder(dataset_path)
    train_merge_patient_notes.to_csv(dataset_path + "/train.csv")
    kaggle_get_metadata(dataset_path, "nbmepldataset")
    kaggle_new_dataset_version(dataset_path)


if __name__ == "__main__":
    make_dataset()
