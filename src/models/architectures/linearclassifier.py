from torch import nn
from transformers import AutoModel


class LinearClassifier(nn.Module):
    """
    Model class that combines a pretrained bert model with a linear later
    """

    def __init__(self, model, num_classes=1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, num_classes),
        )

    def forward(self, inputs, device):
        out = self.transformer(**inputs.to(device))
        logits = self.fc(out[0])
        return logits
