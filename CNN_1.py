# importing the libraries
import os
import cv2
import random
import torch
import torch.nn as nn
import torch.utils.data as td
from prettytable import PrettyTable

EVALUATE_100_IMG = False  # If running on submitted data (100 sample images) then set to true

categories = ["no_mask", "ffp2_mask", "surgical_mask", "cloth_mask"]
img_size = 60
training_data = []


def create_training_data():
    for category in categories:
        path = "./" + category + "/"
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

            new_img = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_img, class_num])


create_training_data()
random.shuffle(training_data)
for data in training_data:
    data[0] = torch.Tensor(data[0])
    data[0] = data[0].permute(2, 0, 1)
    training_data[training_data.index(data)] = tuple(data)

total = len(training_data)
training_percent = .8
if EVALUATE_100_IMG:
    training_percent = .01
train = training_data[:int(total * training_percent)]
train = training_data[:int(total * training_percent)]
test = training_data[int(total * training_percent):]


def cifar_loader(batch_size, shuffle_test=False):
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #     std=[0.225, 0.225, 0.225])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 32
test_batch_size = 32
input_size = 10800
N = batch_size
D_in = input_size
H = 50
D_out = 4
num_epochs = 10
learning_rate = 0.001
train_loader, _ = cifar_loader(batch_size)
_, test_loader = cifar_loader(test_batch_size)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(14400, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, D_out)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


# Initialize model
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# If there is a trained model do not train
has_trained_model = os.path.isfile('cnnsaved.pt')

if has_trained_model:
    model.load_state_dict(torch.load('cnnsaved.pt'), strict=False)
else:
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i + 1) % 15 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    tp_fn_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    tp_fp_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    tp_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    fn_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    fp_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    tn_labels = [0, 0, 0, 0]  # number of occurances for [ "nomask", "ffp2", ... ]
    for images, labels in test_loader:
        outputs = model(images)
        pred_values, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_size = predicted.size()
        for i in range(predicted_size[0]):
            # TP
            if predicted[i].item() == labels[i].item() and predicted[i].item() == 0:
                tp_labels[0] += 1
            elif predicted[i].item() == labels[i].item() and predicted[i].item() == 1:
                tp_labels[1] += 1
            elif predicted[i].item() == labels[i].item() and predicted[i].item() == 2:
                tp_labels[2] += 1
            elif predicted[i].item() == labels[i].item() and predicted[i].item() == 3:
                tp_labels[3] += 1
            # TP + FP
            if predicted[i].item() == 0:
                tp_fp_labels[0] += 1
            elif predicted[i].item() == 1:
                tp_fp_labels[1] += 1
            elif predicted[i].item() == 2:
                tp_fp_labels[2] += 1
            elif predicted[i].item() == 3:
                tp_fp_labels[3] += 1
            # TP + FN
            if labels[i].item() == 0:
                tp_fn_labels[0] += 1
            elif labels[i].item() == 1:
                tp_fn_labels[1] += 1
            elif labels[i].item() == 2:
                tp_fn_labels[2] += 1
            elif labels[i].item() == 3:
                tp_fn_labels[3] += 1

    macro_denom_recall = 4;
    macro_denom_precision = 4;
    recalls = [0, 0, 0, 0]
    precisions = [0, 0, 0, 0]
    f1s = [0, 0, 0, 0]

    for i in range(4):
        # FN
        fn_labels[i] = tp_fn_labels[i] - tp_labels[i]
        # FP
        fp_labels[i] = tp_fp_labels[i] - tp_labels[i]
        # TN
        tn_labels[i] = total - tp_labels[i] - fp_labels[i] - fn_labels[i]
        # precision
        if tp_fp_labels[i] == 0:
            macro_denom_precision -= 1
        else:
            precisions[i] = tp_labels[i] * 100 / tp_fp_labels[i]

        # recall
        if tp_fn_labels[i] == 0:
            macro_denom_recall -= 1
        else:
            recalls[i] = tp_labels[i] * 100 / tp_fn_labels[i]

        # f1
        if (precisions[i] + recalls[i]) == 0:
            pass
        else:
            f1s[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

    precision = sum(precisions) / macro_denom_precision
    recall = sum(recalls) / macro_denom_recall
    accuracy = correct * 100 / total
    f1 = 2 * (precision * recall) / (precision + recall)

    confused_no_mask = PrettyTable()
    confused_no_mask.field_names = [f'No mask (n={total})', "Predicted NO", "Predicted YES"]
    confused_no_mask.add_row(["Actual NO", tn_labels[0], fp_labels[0]])
    confused_no_mask.add_row(["Actual YES", fn_labels[0], tp_labels[0]])
    confused_ffp2 = PrettyTable()
    confused_ffp2.field_names = [f'FFP2 (n={total})', "Predicted NO", "Predicted YES"]
    confused_ffp2.add_row(["Actual NO", tn_labels[1], fp_labels[1]])
    confused_ffp2.add_row(["Actual YES", fn_labels[1], tp_labels[1]])
    confused_cloth = PrettyTable()
    confused_cloth.field_names = [f'Cloth (n={total})', "Predicted NO", "Predicted YES"]
    confused_cloth.add_row(["Actual NO", tn_labels[3], fp_labels[3]])
    confused_cloth.add_row(["Actual YES", fn_labels[3], tp_labels[3]])
    confused_surgical = PrettyTable()
    confused_surgical.field_names = [f'Surgical (n={total})', "Predicted NO", "Predicted YES"]
    confused_surgical.add_row(["Actual NO", tn_labels[2], fp_labels[2]])
    confused_surgical.add_row(["Actual YES", fn_labels[2], tp_labels[2]])

    print(
        f'Overall on {total} images:\taccuracy={round(accuracy, 2)},\tprecision={round(precision, 2)}%,\trecall={round(recall, 2)}%,\tf1 measure={round(f1, 2)}')
    print(confused_no_mask)
    print(
        f'No mask:\tprecision={round(precisions[0], 2)}%,\trecall={round(recalls[0], 2)}%,\tf1 measure={round(f1s[0], 2)}')
    print(confused_ffp2)
    print(
        f'FFP2:\tprecision={round(precisions[1], 2)}%,\trecall={round(recalls[1], 2)}%,\tf1 measure={round(f1s[1], 2)}')
    print(confused_surgical)
    print(
        f'Surgical mask:\tprecision={round(precisions[2], 2)}%,\trecall={round(recalls[2], 2)}%,\tf1 measure={round(f1s[2], 2)}')
    print(confused_cloth)
    print(
        f'Cloth mask:\tprecision={round(precisions[3], 2)}%,\trecall={round(recalls[3], 2)}%,\tf1 measure={round(f1s[3], 2)}')

# Save trained model
torch.save(model.state_dict(), 'cnnsaved.pt')