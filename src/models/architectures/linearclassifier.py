import torch
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


class ThreeHeadsLinearClassifier(nn.Module):
    """
    Model class that combines a pretrained bert model with a linear later
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, 1),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, 1),
        )

    def forward(self, inputs, device):
        out = self.transformer(**inputs.to(device))
        logits1 = self.fc1(out[0])
        logits2 = self.fc2(out[0])
        logits3 = self.fc3(out[0])
        return torch.cat([logits1, logits2, logits3], dim=2)


class TwoHeadsLinearClassifier(nn.Module):
    """
    Model class that combines a pretrained bert model with a linear later
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.transformer.config.hidden_size, 1),
        )

    def forward(self, inputs, device):
        out = self.transformer(**inputs.to(device))
        logits1 = self.fc1(out[0])
        logits2 = self.fc2(out[0])
        return torch.cat([logits1, logits2], dim=2)
