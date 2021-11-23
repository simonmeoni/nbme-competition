import os
import shutil

from bin.checkpoints.upload_to_kaggle import kaggle_get_metadata, kaggle_new_dataset_version
from bin.transformers.download_huggingface_model import download_huggingface_model

kaggle_dataset = "pretrainedmodels"
folder = "./models/"
pt_path = "../models/pt"
if os.path.exists(pt_path):
    shutil.rmtree(pt_path, ignore_errors=False)
os.makedirs(pt_path)


download_huggingface_model(pt_path, "roberta-base")
download_huggingface_model(pt_path, "microsoft/deberta-v3-base")
download_huggingface_model(pt_path, "unitary/unbiased-toxic-roberta")
download_huggingface_model(pt_path, "vinai/bertweet-base")
download_huggingface_model(pt_path, "unitary/toxic-bert")
download_huggingface_model(pt_path, "google/electra-base-discriminator")
download_huggingface_model(pt_path, "xlnet-base-cased")
download_huggingface_model(pt_path, "microsoft/deberta-v3-large")
download_huggingface_model(pt_path, "roberta-large")
download_huggingface_model(pt_path, "google/electra-large-discriminator")
download_huggingface_model(pt_path, "xlnet-large-cased")
download_huggingface_model(pt_path, "distilroberta-base")
download_huggingface_model(pt_path, "yikuan8/Clinical-Longformer")
download_huggingface_model(pt_path, "emilyalsentzer/Bio_ClinicalBERT")
download_huggingface_model(pt_path, "microsoft/deberta-v2-xlarge")
download_huggingface_model(pt_path, "google/electra-large-generator")
download_huggingface_model(pt_path, "gpt2")

download_huggingface_model(pt_path, "smeoni/nbme-Bio_ClinicalBERT")
download_huggingface_model(pt_path, "smeoni/nbme-deberta-large")
download_huggingface_model(pt_path, "smeoni/nbme-roberta-large")
download_huggingface_model(pt_path, "smeoni/nbme-deberta-v3-large")
download_huggingface_model(pt_path, "smeoni/nbme-electra-large-discriminator")
download_huggingface_model(pt_path, "smeoni/nbme-xlnet-large-cased")
download_huggingface_model(pt_path, "smeoni/nbme-deberta-v2-xlarge")
download_huggingface_model(pt_path, "smeoni/nbme-electra-large-generator")
download_huggingface_model(pt_path, "smeoni/nbme-gpt2")
download_huggingface_model(pt_path, "smeoni/nbme-clinical-longformer")


kaggle_get_metadata(pt_path, kaggle_dataset)
kaggle_new_dataset_version(pt_path)
