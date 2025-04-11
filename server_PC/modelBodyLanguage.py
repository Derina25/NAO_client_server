import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class bodyLanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(bodyLanguageClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify final layers
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze initial layers
        for name, param in self.model.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

# Esempio di utilizzo del modello
num_classes = 6  # Numero di classi nel tuo dataset
model = bodyLanguageClassifier(num_classes)