import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import torchvision
# from torchvision import datasets, transforms, models
from collections import Counter
import tqdm, math
import pandas as pd

# 早期停止
# 早期停止类有助于根据验证损失跟踪最佳模型，并保存检查点。
# Callbacks
# Early stopping
class EarlyStopping:
    def __init__(self, patience=1, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class CassavaClassifier():
    def __init__(self, data_dir, num_classes, device, Transform=None, sample=False, loss_weights=False, batch_size=16,
     lr=1e-4, stop_early=True, freeze_backbone=True):
    #############################################################################################################
    # data_dir - directory with images in subfolders, subfolders name are categories
    # Transform - data augmentations
    # sample - if the dataset is imbalanced set to true and RandomWeightedSampler will be used
    # loss_weights - if the dataset is imbalanced set to true and weight parameter will be passed to loss function
    # freeze_backbone - if using pretrained architecture freeze all but the classification layer
    ###############################################################################################################
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.device = device
        self.sample = sample
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.lr = lr
        self.stop_early = stop_early
        self.freeze_backbone = freeze_backbone
        self.Transform = Transform

    def load_data(self):
        """
        Load Data
        训练图像被组织在子文件夹中，子文件夹名称表示图像的类。
        这是图像分类问题的典型情况，幸运的是，不需要编写自定义数据集类。
        在这种情况下，可以立即使用torchvision中的ImageFolder。
        如果你想使用WeightedRandomSampler，你需要为数据集的每个元素指定一个权重。
        通常，总图像总比上类别数被用作一个权重。
        :return:
        """
        train_full = torchvision.datasets.ImageFolder(self.data_dir, transform=self.Transform)
        train_set, val_set = random_split(train_full,
                                          [math.floor(len(train_full) * 0.8), math.ceil(len(train_full) * 0.2)])

        self.train_classes = [label for _, label in train_set]
        if self.sample:
            # Need to get weight for every image in the dataset
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor(
                [len(self.train_classes) / c for c in pd.Series(class_count).sort_index().values])
            # Can't iterate over class_count because dictionary is unordered
            sample_weights = [0] * len(train_set)
            for idx, (image, label) in enumerate(train_set):
                class_weight = class_weights[label]
                sample_weights[idx] = class_weight
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(train_set), replacement=True)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler)
        else:
            # 核心部分为实际使用时替换下变量把sampler传递给DataLoader即可，注意使用了sampler就不能使用shuffle，另外需要指定采样点个数。
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        return train_loader, val_loader

    def load_model(self, arch='resnet'):
        """
        Load Model
        在该方法中，我使用迁移学习，架构参数从预先训练的resnet50和efficientnet-b7中选择。
        CrossEntropyLoss和许多其他损失函数都有权重参数。
        这是一个手动调整参数，用于处理不平衡。
        在这种情况下，不需要为每个参数定义权重，只需为每个类定义权重。
        :param arch:
        :return:
        """
        ##############################################################################################################
        # arch - choose the pretrained architecture from resnet or efficientnetb7
        ##############################################################################################################
        if arch == 'resnet':
            self.model = torchvision.models.resnet50(pretrained=True)
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        elif arch == 'efficient-net':
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=self.num_classes)

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor(
                [len(self.train_classes) / c for c in pd.Series(class_count).sort_index().values])
            # Cant iterate over class_count because dictionary is unordered
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def fit_one_epoch(self, train_loader, epoch, num_epochs):
        """
        Fit One Epoch
        这个方法只包含一个经典的训练循环，带有训练损失记录和tqdm进度条。
        :param train_loader:
        :param epoch:
        :param num_epochs:
        :return:
        """
        step_train = 0

        train_losses = list()  # Every epoch check average loss per batch
        train_acc = list()
        self.model.train()
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()

            train_losses.append(loss.item())

            # Calculate running train accuracy
            predictions = torch.argmax(logits, dim=1)
            num_correct = sum(predictions.eq(targets))
            running_train_acc = float(num_correct) / float(images.shape[0])
            train_acc.append(running_train_acc)

        train_loss = torch.tensor(train_losses).mean()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Training loss: {train_loss:.2f}')

    def val_one_epoch(self, val_loader, scaler):
        """
        Validate one epoch
        与上面类似，但此方法在验证数据加载器上迭代。
        在每一个epoch'之后，平均batch损失和准确性被打印出来。
        :param val_loader:
        :param scaler:
        :return:
        """
        val_losses = list()
        val_accs = list()
        self.model.eval()
        step_val = 0
        with torch.no_grad():
            for (images, targets) in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                val_losses.append(loss.item())

                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_val_acc = float(num_correct) / float(images.shape[0])

                val_accs.append(running_val_acc)

            self.val_loss = torch.tensor(val_losses).mean()
            val_acc = torch.tensor(val_accs).mean()  # Average acc per batch

            print(f'Validation loss: {self.val_loss:.2f}')
            print(f'Validation accuracy: {val_acc:.2f}')

    def fit(self, train_loader, val_loader, num_epochs=10, unfreeze_after=5, checkpoint_dir='checkpoint.pt'):
        """
        Fit
        Fit方法在训练和验证过程中经历了许多阶段和循环。
        如果预训练模型的参数在开始时被冻结，那么unfreeze_after定义了整个模型在多少个epoch之后开始训练。
        在此之前，只训练全连接层(分类器)。
        :param train_loader:
        :param val_loader:
        :param num_epochs:
        :param unfreeze_after:
        :param checkpoint_dir:
        :return:
        """
        if self.stop_early:
            early_stopping = EarlyStopping(
                patience=5,
                path=checkpoint_dir)

        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:  # Unfreeze after x epochs
                    for param in self.model.parameters():
                        param.requires_grad = True
            self.fit_one_epoch(train_loader, scaler, epoch, num_epochs)
            self.val_one_epoch(val_loader, scaler)
            if self.stop_early:
                early_stopping(self.val_loss, self.model)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    print(f'Best validation loss: {early_stopping.best_score}')
                    break

if __name__ == "__main__":
    # Run
    # 现在，可以初始化CassavaClassifier类、创建dataloaders、设置模型并运行整个过程了。

    import torchvision.transforms as T

    Transform = T.Compose(
        [T.ToTensor(),
         T.Resize((256, 256)),
         T.RandomRotation(90),
         T.RandomHorizontalFlip(p=0.5),
         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = "Data/cassava-disease/train/train"

    classifier = CassavaClassifier(data_dir=data_dir, num_classes=5, device=device, sample=True, Transform=Transform)
    train_loader, val_loader = classifier.load_data()
    classifier.load_model()
    classifier.fit(num_epochs=20, unfreeze_after=5, train_loader=train_loader, val_loader=val_loader)

    #########################################################################################
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import os
    # Inference
    # 使用ImageFolder加载测试数据是不可能的，因为显然没有带有类的子文件夹。
    # 因此，我创建了一个返回图像和图像id的自定义数据集。
    # 随后，加载模型检查点，通过推理循环运行它，并将预测保存到数据帧中。
    # 将数据帧导出为CSV并提交结果。
    # Inference
    model = torchvision.models.resnet50()
    # model = EfficientNet.from_name('efficientnet-b7')
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)
    model = model.to(device)
    checkpoint = torch.load('Data/cassava-disease/sampler_checkpoint.pt')
    model.load_state_dict(checkpoint)
    model.eval()


    # Dataset for test data
    class Cassava_Test(Dataset):
        def __init__(self, dir, transform=None):
            self.dir = dir
            self.transform = transform

            self.images = os.listdir(self.dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.dir, self.images[idx]))
            return self.transform(img), self.images[idx]


    test_dir = 'Data/cassava-disease/test/test/0'
    test_set = Cassava_Test(test_dir, transform=Transform)
    test_loader = DataLoader(test_set, batch_size=4)

    # Test loop
    sub = pd.DataFrame(columns=['category', 'id'])
    id_list = []
    pred_list = []

    model = model.to(device)

    with torch.no_grad():
        for (image, image_id) in test_loader:
            image = image.to(device)

            logits = model(image)
            predicted = list(torch.argmax(logits, 1).cpu().numpy())

            for id in image_id:
                id_list.append(id)

            for prediction in predicted:
                pred_list.append(prediction)
    sub['category'] = pred_list
    sub['id'] = id_list

    mapping = {0: 'cbb', 1: 'cbsd', 2: 'cgm', 3: 'cmd', 4: 'healthy'}

    sub['category'] = sub['category'].map(mapping)
    sub = sub.sort_values(by='id')

    sub.to_csv('Cassava_sub.csv', index=False)