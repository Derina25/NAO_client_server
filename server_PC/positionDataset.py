import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class catPositionsDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Definisci le trasformazioni di data augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        # Rimuovi i dati EXIF (img corrotta)
        image.info.pop('exif', None)

        if self.label_dir:
            label_path = os.path.join(self.label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            boxes = self.load_yolo_labels(label_path)
        else:
            boxes = torch.tensor([])  # Se non ci sono etichette, restituisci un tensore vuoto

        # Applica le trasformazioni di data augmentation
        if self.augmentation:
            image = self.augmentation(image)

        if self.transform:
            image = self.transform(image)

        return image, boxes

    def load_yolo_labels(self, label_path):
        boxes = []
        try:
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])
            return torch.tensor(boxes, dtype=torch.float32)
        except FileNotFoundError:
            print(f"File not found: {label_path}")
            return torch.tensor([], dtype=torch.float32)
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")
            return torch.tensor([], dtype=torch.float32)

    @property
    def classes(self):
        return ['accovacciato', 'alzato', 'seduto', 'steso'] 