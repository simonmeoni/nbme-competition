import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
from flair.models import MultiTagger
from flair.tokenization import SciSpacySentenceSplitter
from tqdm import tqdm

from bin.checkpoints.upload_to_kaggle import kaggle_get_metadata, kaggle_new_dataset_version
from bin.file_utils.rm_new_folder import rm_and_new_folder


def make_dataset():
    train = pd.read_csv("data/nbme-score-clinical-patient-notes/train.csv")
    features = pd.read_csv("data/nbme-score-clinical-patient-notes/features.csv")
    patient_notes = pd.read_csv("data/nbme-score-clinical-patient-notes/patient_notes.csv")
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
    augment_data(train_merge_patient_notes)
    dataset_path = "data/nbme-merge-dataset"
    rm_and_new_folder(dataset_path)
    train_merge_patient_notes.to_csv(dataset_path + "/train.csv")
    kaggle_get_metadata(dataset_path, "nbme-merge-dataset")
    kaggle_new_dataset_version(dataset_path)


def augment_data(dataset):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name="facebook/wmt19-en-de", to_model_name="facebook/wmt19-de-en", device="cuda"
    )
    locations = dataset.location.apply(
        lambda x: [
            [int(numeric) for numeric in offset.split(" ")]
            for loc in eval(x)
            for offset in loc.split(";")
        ]
    )
    augmented_locations_list = []
    augmented_examples_list = []
    tk0 = tqdm(dataset.iterrows(), total=len(dataset))
    for idx, row in tk0:
        note = row.pn_history
        chunks = []
        previous_right_offset = 0
        augmented_locations = []
        if len(locations[idx]) != 0:
            for loc in locations[idx]:
                left = note[previous_right_offset : loc[0]]
                previous_right_offset = loc[1]
                chunks.append(back_translation(back_translation_aug, left))
            chunks.append(back_translation(back_translation_aug, note[previous_right_offset:]))
            augmented_pn_history = ""
            for loc, chunk in zip(locations[idx], chunks[:-1]):
                augmented_pn_history += chunk
                start = len(augmented_pn_history)
                end = start + (loc[1] - loc[0])
                augmented_locations.append([start, end])
                augmented_pn_history += note[loc[0] : loc[1]]
            augmented_pn_history += chunks[-1]
            augmented_examples_list.append(augmented_pn_history)
        else:
            augmented_examples_list.append(back_translation(back_translation_aug, note))
        augmented_locations_list.append(augmented_locations)
    dataset["augmented_examples"] = augmented_examples_list
    dataset["augmented_locations"] = augmented_locations_list


def back_translation(back_translation_aug, left):
    return "\n".join([back_translation_aug.augment(line) for line in left.split("\n")])


def get_new_entities(patient_notes):
    splitter = SciSpacySentenceSplitter()
    tagger = MultiTagger.load("hunflair")
    patient_notes["diseases_loc"] = np.empty((len(patient_notes), 0)).tolist()
    patient_notes["chemicals_loc"] = np.empty((len(patient_notes), 0)).tolist()
    tk0 = tqdm(patient_notes.iterrows(), total=len(patient_notes))
    for _, note in tk0:
        sentences = splitter.split(note.pn_history)
        tagger.predict(sentences)
        sentence_offset = 0
        for sentence in sentences:
            for span in sentence.get_spans():
                label = span.get_labels()[0].value
                span_offset = [
                    span.start_pos + sentence_offset,
                    span.end_pos + sentence_offset,
                ]
                if label == "Disease":
                    note.diseases_loc.append(span_offset)
                elif label == "Chemical":
                    note.chemicals_loc.append(span_offset)
            sentence_offset += len(sentence.to_plain_string())


if __name__ == "__main__":
    make_dataset()
