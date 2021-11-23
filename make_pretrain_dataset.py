import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download("punkt")


def create_line_dataset(output, dataset):
    with open(output, "w") as file:
        for _, line in dataset.iterrows():
            sentences = line.pn_history.replace("\n", " ").replace("\r", " ").strip() + "\n"
            for sentence in nltk.sent_tokenize(sentences):
                if len(sentence.split(" ")) > 20:
                    file.write(sentence + "\n")


if __name__ == "__main__":
    patient_notes = pd.read_csv("data/nbme-score-clinical-patient-notes/patient_notes.csv")
    train_dataset, eval_dataset = train_test_split(patient_notes, test_size=0.15)
    create_line_dataset(dataset=train_dataset, output="data/train.txt")
    create_line_dataset(dataset=eval_dataset, output="data/eval.txt")
