import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

# --- 1. Настройка трансформаций ---
# Для обучения добавляем случайное отражение, чтобы уменьшить переобучение
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Для теста только базовые изменения
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# --- 2. Класс Датасета ---
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Предупреждение: Папка {class_dir} не найдена.")
                continue
            for file in glob.glob(os.path.join(class_dir, "*.jpg")):
                self.image_paths.append(file)
                self.labels.append(self.class_to_idx[class_name])
        print(f"Успешно загружено {len(self.image_paths)} изображений из {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Используем .convert('L') для гарантии одного канала (Grayscale)
        image = Image.open(self.image_paths[index]).convert('L')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 3. Архитектура нейросети (CNN) ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Слой 1: Вход 1 канал (серый), выход 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Слой 2: Выход 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Слой 3: Выход 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        
        # Автоматический расчет размера после сверток
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.relu_fc = nn.ReLU()
        self.drop_out = nn.Dropout(0.5) # Защита от переобучения
        self.fc2 = nn.Linear(256, 4) # 4 класса опухолей

    def _get_conv_output(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 256, 256)
            dummy = self.layer1(dummy)
            dummy = self.layer2(dummy)
            dummy = self.layer3(dummy)
            self._to_linear = dummy.view(1, -1).size(1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out

# --- 4. Функции для обучения ---
def train_epoch(model, loader, loss_func, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

def validate_epoch(model, loader, loss_func, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Обучение (Loss)')
    plt.plot(history['val_loss'], label='Валидация (Loss)')
    plt.title('История потерь')
    plt.xlabel('Эпоха')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Обучение (Acc)')
    plt.plot(history['val_acc'], label='Валидация (Acc)')
    plt.title('История точности')
    plt.xlabel('Эпоха')
    plt.legend()
    plt.show()

# --- 5. Запуск процесса ---
if __name__ == "__main__":
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    # Инициализация данных
    # ЗАМЕНИ ПУТИ, ЕСЛИ ОНИ ОТЛИЧАЮТСЯ
    train_dataset = BrainTumorDataset(r"training_data\Training", train_transform)
    test_dataset = BrainTumorDataset(r"training_data\Testing", test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Модель, оптимизатор и функция потерь
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    
    num_epochs = 20
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print("Начинаем обучение...")
    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, loss_func, optimizer, device)
        v_loss, v_acc = validate_epoch(model, test_loader, loss_func, device)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Эпоха {epoch+1}/{num_epochs}: Train Acc: {t_acc:.2f}%, Val Acc: {v_acc:.2f}%")

        # Сохраняем только лучшую версию весов
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), 'best_brain_model.pth')
            print(f"--- Модель сохранена (Accuracy: {v_acc:.2f}%) ---")

    print(f"\nОбучение завершено! Лучшая точность: {best_acc:.2f}%")
    plot_history(history)