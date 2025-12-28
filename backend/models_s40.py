import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

s40_classes = [
    "applauding","blowing_candles","brushing_teeth","cleaning_floor","climbing","cooking","cutting_vegetables",
    "drinking","fishing","fixing_hair","gardening","holding_an_umbrella","jumping","looking_through_microscope",
    "looking_through_telescope","phoning","playing_violin","pouring_liquid","pushing","reading","writing_on_board",
    "riding_bike","riding_horse","running","shooting_an_arrow","smoking","taking_photos","throwing_frisby",
    "using_computer","walking_the_dog","waving_hands","answering_questions","tying_tie","ironing","feeding_a_baby",
    "wiping_faces","hula_hoop","jogging","texting_message","feeding_a_pet"
]

class LSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        y, _ = self.lstm(x)
        h = y[:, -1, :]
        return self.fc(h)

class S40ResNet18LSTM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = LSTMHead(512, 256, num_classes)
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        f = self.features(x)
        f = f.view(b, t, -1)
        return self.head(f)

def build_s40_model(num_classes: int) -> nn.Module:
    return S40ResNet18LSTM(num_classes)

s40_image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
