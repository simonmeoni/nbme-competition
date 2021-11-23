import pandas as pd

from datamodules.datamodule import NBMEDataset


def test_create_labels():
    """Test running for 1 train, val and test batch."""

    tokenizer_config = {
        "max_length": 512,
        "padding": "longest",
        "return_offsets_mapping": True,
        "return_tensors": "pt",
    }

    full_dataset = NBMEDataset(
        pd.read_csv("data/nbme-merge-dataset/train.csv").dropna(),
        "distilroberta-base",
        tokenizer_config=tokenizer_config,
    )
    text = [[p, f] for p, f in zip(full_dataset.pn_history, full_dataset.feature_text)]
    tokens = full_dataset.tokenizer(text, **tokenizer_config)
    full_dataset.create_labels(tokens, [loc for loc in full_dataset.location])
