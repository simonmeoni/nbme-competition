from torch import nn
from transformers import AutoModel


class LSTM(nn.Module):
    """
    Model class that combines a pretrained bert model with a gru later
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.gru = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=768,
            batch_first=True,
            bidirectional=True,
            num_layers=8,
            dropout=0.1,
        )
        self.linear = nn.Linear(768 * 2, 1)

    def forward(self, inputs, device):
        out = self.transformer(**inputs.to(device))
        logits = self.gru(out[0])
        return self.linear(logits[0])
