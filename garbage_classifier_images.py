# import libraries
import torch
import glob # to obtain file paths for image dataset
import matplotlib.pylab as plt # for plotting example data
from torch.utils.data import Dataset, DataLoader # torch dataset and dataloader
import torch.nn as nn
import numpy as np # work with arrays
from PIL import Image # read an image from file system
from torchvision.models import resnet34 # base model for transfer learning
from torchvision.models import ResNet34_Weights # contains the transforms used on original data
from torchvision import transforms, models # contains transforms for data augmentation
from sklearn.model_selection import StratifiedShuffleSplit # to shuffle our imbalanced dataset
from torch import optim # optimizer
from torch.optim.lr_scheduler import ExponentialLR # learning rate scheduler
import os 
import pandas as pd # for dataframes
import datetime as dt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn

print(dt.datetime.now(), "Import modules completed.")
print(dt.datetime.now(), "Preprocessing images starting.")

# 1. PREPROCESS IMAGES

def add_margin(height, width, pil_image, top, right, bottom, left, color):
    """
    Utility function to paste the original image on top of a new black image with a margin.
    The result is a new SQUARE image with margins added to the smaller side 
    on either side of the original image to make it square.
    
    Parameters
    ----------
    height: double
        original image height
    width: double
        original image width
    pil_image: PIL.Image
        the original image opened with PIL
    top: double
        number of pixels to pad the top
    right: double
        number of pixels to pad the right
    bottom: double
        number of pixels to pad the bottom
    left: double
        number of pixels to pad the left
    color: tuple
        color to use for the image
    """
    
    new_width = width + right + left
    new_height = height + top + bottom
    # create a new image with new_width, new_height dimensions in the
    # same mode as the original image
    result = Image.new(pil_image.mode, (new_width, new_height), color)
    # paste the original image onto the new black image displaced from
    # the left and top by the specified number of pixels
    result.paste(pil_image, (left, top))
    return result

def keep_AR(image):
    """
    Utility function to calculate how much margin to add to each side of an image
    to make it square.
    
    image: PIL.Image
        the original image opened with PIL
    """
    target_aspect_ratio = 1
    original_width = image.width
    original_height = image.height
    current_aspect_ratio = original_width / original_height
    new_image = []
    
    if current_aspect_ratio == target_aspect_ratio:
        new_image = image
    if current_aspect_ratio < target_aspect_ratio:
        # need to increase width
        target_width = int(target_aspect_ratio * original_height)
        pad_amount_pixels = target_width - original_width
        new_image = add_margin(original_height, original_width, image, 0, int(pad_amount_pixels/2), 0, int(pad_amount_pixels/2), (0, 0, 0))
    if current_aspect_ratio > target_aspect_ratio:
        # need to increase height
        target_height = int(original_width/target_aspect_ratio)
        pad_amount_pixels = target_height - original_height
        new_image = add_margin(original_height, original_width, image, int(pad_amount_pixels/2), 0, int(pad_amount_pixels/2), 0, (0, 0, 0))

    return new_image
        
def resize(class_):
    """
    Utility function to resize all images in a folder to square 512 x 512 images
    
    class_: str
        name of the folder to resize images
    """
    # path to the folder with the class folders, black, blue, green, other
    path = "../dataset/" + class_
    # get list of all image names
    dirs = os.listdir(path)
    count = 0
    # new folder name with resized images
    resized_path = class_ + "_resized"
    for item in dirs:
        print(path + "/" + item)
        if os.path.isfile(path + "/" + item):
            try:
                image = Image.open(path + "/" + item)
                # calculate what margin to add to the left/right or top/bottom
                # and then create a new image, paste the original image onto the new
                # image displaced by the left/right and top/bottom margins
                image = keep_AR(image)
                image_resized = image.resize((512, 512), Image.ANTIALIAS)
                if not os.path.exists("../dataset/" + resized_path + "/"):
                    os.mkdir("../dataset/" + resized_path + "/")
                image_resized.convert("RGB").save(f"../dataset/" + resized_path + "/" + class_ + "_" + str(count) + "_rgba.jpg")
                count += 1
            except Exception as e:
                print(e)
            
# Preprocess the extra data obtained from Prof Souza to 512 x 512 x 3 jpg images
# do not resize the images more than once
print()
resize_images = False
if resize_images:
    resize("black")
    resize("blue")
    resize("green")
    resize("other")

if resize_images:
    print(dt.datetime.now(), "Preprocess images completed.")
else:
    print(dt.datetime.now(), "Preprocess images skipped.")
print(dt.datetime.now(), "Loading the dataset starting.")

# 2. LOAD THE DATASET

# Load the dataset
filepath_to_dataset = "/home/lplee/enel_645_finalproject/images/"

image_paths = glob.glob(filepath_to_dataset + "*/*.jpg")

# full path to each image
image_paths = np.array(image_paths)

# labels for each image as a string
labels_string = np.array([f.split("/")[-2] for f in image_paths])

print(f"Number of images is {len(image_paths)}")
print(f"Number of labels is {len(labels_string)}")
print(f"Example image_path is {image_paths[0]}")
print(f"Example label is {labels_string[0]}")

# unqiue class names
classes = np.unique(labels_string).flatten()
print(f"Classes are {classes}")

# convert our labels_string vector to a integer vector
labels_int = np.zeros(labels_string.size, dtype=np.int64)
for index, class_name in enumerate(classes):
    labels_int[labels_string == class_name] = index

print(dt.datetime.now(), "Loading the dataset completed.")
print(dt.datetime.now(), "Creating class_distribution.png")

# 3. EXPLORATORY DATA ANALYSIS

# plot the distribution of the classes 
plt.figure(figsize=(10, 10))
df = pd.DataFrame(labels_int, columns=["Class"])
class_counts = df["Class"].value_counts().sort_index()
plt.bar(classes, class_counts.values)
plt.title("Distribution of samples")
plt.xlabel("Bin Color")
plt.ylabel("Count of Samples")
plt.savefig("class_distribution.png", transparent=False, facecolor='white')

print(dt.datetime.now(), "Created class_distribution.png")
print(dt.datetime.now(), "Splitting the data into train, val, test starting.")

# 4. SPLIT INTO TRAIN, VAL, TEST

# split image_paths and labels_int into development and test set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
sss.get_n_splits(image_paths, labels_int)
dev_index, test_index = next(sss.split(image_paths, labels_int))

# development set image paths and labels
dev_image_paths = image_paths[dev_index]
dev_labels_int = labels_int[dev_index]

# test set image path and labels
test_image_paths =  image_paths[test_index]
test_labels_int = labels_int[test_index]

# split dev_image_paths and dev_labels_int into validation and training set
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
sss2.get_n_splits(dev_image_paths, dev_labels_int)
train_index, val_index = next(sss.split(dev_image_paths, dev_labels_int))

#training set image paths and labels
train_image_paths = dev_image_paths[train_index]
train_labels_int = dev_labels_int[train_index]

# validation set image paths and labels
val_image_paths = dev_image_paths[val_index]
val_labels_int = dev_labels_int[val_index]

print(f"Train set")
print(f"Images size {train_image_paths.size}, shape is {train_image_paths.shape}")
print(f"Labels size is {train_labels_int.size}, shape is {train_labels_int.shape}")
print()
print(f"Validation set")
print(f"Images size {val_image_paths.size}, shape is {val_image_paths.shape}")
print(f"Labels size is {val_labels_int.size}, shape is {val_labels_int.shape}")
print()
print(f"Test set")
print(f"Images size {test_image_paths.size}, shape is {test_image_paths.shape}")
print(f"Labels size is {test_labels_int.size}, shape is {test_labels_int.shape}")
print()

print(dt.datetime.now(), "Splitting the data into train, val, test completed.")
print(dt.datetime.now(), "Saving sample images from the training set in sample_images.png.")

# 5. EXPLORATORY DATA ANALYSIS

# displaying some samples from the training set
sample_indexes = np.random.choice(np.arange(train_image_paths.shape[0], dtype=int), size=30, replace=False)
plt.figure(figsize=(24,18))
for (index, sample_index) in enumerate(sample_indexes):
    plt.subplot(5, 6, index+1)
    plt.imshow(Image.open(train_image_paths[sample_index]))
    plt.title(f"Label: {classes[train_labels_int[sample_index]]}")
plt.savefig("sample_images.png", facecolor="white")

print(dt.datetime.now(), "Created sample_images.png.")
print(dt.datetime.now(), "Defining dataset and dataloaders.")

# 6. DEFINE DATASET CLASS AND DATALOADERS

class TorchVisionDataset(Dataset):
    """
    Dataset class that defines how to obtain a sample
    for our data loader.
    """
    
    def __init__(self, data_dic, transform=None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        image = Image.open(file_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# define dictionaries for each set with X as the filepaths and Y as the labels
train_set = {
    "X": train_image_paths,
    "Y": train_labels_int
}

val_set = {
    "X": val_image_paths,
    "Y": val_labels_int
}

test_set = {
    "X": test_image_paths,
    "Y": test_labels_int
}

print(dt.datetime.now(), "Calculating mean and std for each channel in the train set starting.")

# define a function to calculate the mean and std for each channel of the images in the train set
def get_dataset_stats(dataset_image_paths):
    """
    Utility function to calculate the mean and std of each channel
    in a set of images.
    
    dataset_image_paths: list[str]
        list of file paths to images
    """
    mean = 0
    std = 0
    nb_samples = 0
    for image_path in dataset_image_paths:
        # get the image to compute the statistics
        image = np.array(Image.open(image_path))
        image = image / 255
        mean += np.mean(image, axis=(0, 1))
        std += np.std(image, axis=(0, 1))
        nb_samples += 1
        
    mean /= nb_samples
    std /= nb_samples

    return mean, std

# do not calculate the statistics when training
get_train_stats = True
if get_train_stats:
    train_set_means, train_set_stds = get_dataset_stats(train_image_paths)

    print(f"The means for the RGB channels in the train set is \n{train_set_means}")
    print(f"The stds for the RGB channels in the train set is \n{train_set_stds}")

if get_train_stats:
    print(dt.datetime.now(), "Calculating mean and std for each channel in the train set completed.")
else:
    print(dt.datetime.now(), "Calculating mean and std for each channel in the train set skipped.")

# define the transforms for the train and validation set for the data loader
torchvision_transforms_train_val = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((0, 180)),
    transforms.ToTensor(),
    ResNet34_Weights.IMAGENET1K_V1.transforms() # apply the same transforms that the original model does
])

# define the transforms for the test set for the data loader
torchvision_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    ResNet34_Weights.IMAGENET1K_V1.transforms() # apply the same transforms that the original model does
])

# create a dataset for train, val and test
train_dataset = TorchVisionDataset(train_set, transform=torchvision_transforms_train_val)
val_dataset = TorchVisionDataset(val_set, transform=torchvision_transforms_train_val)
test_dataset = TorchVisionDataset(test_set, transform=torchvision_transforms_test)

# create the data loaders for train, val and test
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0)

print(dt.datetime.now(), "Datasets, Dataloaders defined completed.")
print(dt.datetime.now(), "Saving augmented sample images from the training set in sample_images_augmented.png.")

# 7. VIEW AUGMENTED SAMPLES

# displaying some samples from the training set from the DataLoader
train_iterator = iter(train_loader)
train_batch = next(train_iterator)

print("Train batch samples shape")
print(f"{train_batch[0].size()}")
print("Train batch labels shape")
print(f"{train_batch[1].size()}")

# sample_indexes = np.random.choice(np.arange(train_image_paths.shape[0], dtype=int), size=30, replace=False)
plt.figure(figsize=(24,18))
for (index, sample) in enumerate(train_batch[0][:-2]):
    plt.subplot(5, 6, index+1)
    plt.imshow(sample.numpy().transpose(1, 2, 0))
    plt.title(f"Label: {classes[train_batch[1][index]]}")
plt.savefig("sample_images_augmented.png", facecolor="white")
plt.show()

print(dt.datetime.now(), "Created sample_images_augmented.png.")
print(dt.datetime.now(), "Defining neural network model starting.")

# 8. DEFINE NEURAL NETWORK

class GarbageModel(nn.Module):
    """
    Neural network defined using the subclassing API
    """
    
    def __init__(self, num_classes, input_shape, transfer=True):
        
        super().__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # transfer learning if pretrained=True
        self.base_model = models.resnet34(pretrained=transfer)
        
        if self.transfer:
            # freeze layers using eval()
            self.base_model.eval()
            
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        n_features = self._get_conv_output(self.input_shape)
        self.classifier = nn.Linear(n_features, num_classes)
        
    def _get_conv_output(self, shape):
        
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.base_model(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def forward(self, x):
        
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
# check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(dt.datetime.now(), "Checking if we are on a CUDA machine.")
print(device) 

print(dt.datetime.now(), "Defining neural network model completed.")
print(dt.datetime.now(), "Sending neural network to device starting.")

# create our neural network model and send to device
net = GarbageModel(4, (3, 224, 224), True)
net.to(device)

print(dt.datetime.now(), "Defining neural network model completed.")
print(dt.datetime.now(), "Defining criterion, optimizer, scheduler and starting training loop.")

# define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# training loop
nepochs = 50
PATH = "./garbage_net.pth" # save path for the best model

best_loss = 1e+20
prev_loss = 1e+20
PATIENCE = 20
counter = 0
for epoch in range(nepochs):  # epoch loop
    # Training Loop
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0): # batch loop
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'{dt.datetime.now()} {epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
    scheduler.step()
    
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
        print(f'{dt.datetime.now()} val loss: {val_loss / i:.3f}')
        
        # Save best model
        if val_loss < best_loss:
            print(f"{dt.datetime.now()} Saving model")
            torch.save(net.state_dict(), PATH)
            best_loss = val_loss
        
        # Check for early-stopping
        if val_loss > prev_loss:
            counter += 1
            if counter > PATIENCE:
                print(f"{dt.datetime.now()} Early stopping")
                break
        else:
            counter = 0
        prev_loss = val_loss
        
print(f'{dt.datetime.now()} Finished Training')

print(f'{dt.datetime.now()} Loading the best model, unfreezing the layers, and training for a few more epochs.')

# Load the best model, unfreeze the base_model, train for a few epochs
net = GarbageModel(4, (3, 224, 224), True)
net.load_state_dict(torch.load(PATH))
net.to(device)

# unfreeze the base_model
for param in net.base_model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr = 1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)

nepochs = 10
PATH = "./garbage_net.pth" # save path for the best model

best_loss = 1e+20
prev_loss = 1e+20
PATIENCE = 20
counter = 0
for epoch in range(nepochs):  # epoch loop
    # Training Loop
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0): # batch loop
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'{dt.datetime.now()} {epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
    scheduler.step()
    
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
        print(f'{dt.datetime.now()} val loss: {val_loss / i:.3f}')
        
        # Save best model
        if val_loss < best_loss:
            print(f"Saving model")
            torch.save(net.state_dict(), PATH)
            best_loss = val_loss
        
        # Check for early-stopping
        if val_loss > prev_loss:
            counter += 1
            if counter > PATIENCE:
                print(f"{dt.datetime.now()} Early stopping")
                break
        else:
            counter = 0
        prev_loss = val_loss
        
print(f'{dt.datetime.now()} Finished Training')

print(f'{dt.datetime.now()} Loading the best model and evaluating it on the test set.')

# Load the best model to be used in the test set
net = GarbageModel(4, (3, 224, 224), True)
net.load_state_dict(torch.load(PATH))

# 9. ANALYSE TEST RESULTS

correct = 0
total = 0
figure_name_count = 0

# Label arrays for confusion matrix (next step)
y_true = []
y_pred = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        y_true.extend(labels.data.cpu().numpy())
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.data.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Display some incorrect predictions
        if (predicted != labels).sum().item() > 0 and random.randint(0, 100) < 2:
            incorrect_images = images[predicted != labels]
            incorrect_labels = labels[predicted != labels]
            incorrect_predicted = predicted[predicted != labels]
            
            plt.figure()
            plt.imshow(incorrect_images[0].numpy().transpose(1, 2, 0))
            plt.title(f"Label: {classes[incorrect_labels[0]]}, Predicted: {classes[incorrect_predicted[0]]}")            
            plt.savefig(f"incorrect_{figure_name_count}", facecolor='white')
            figure_name_count += 1
            plt.show()
print(f'{dt.datetime.now()} Accuracy of the network on the test images: {100 * correct / total} %')

# Display and save confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cf_matrix = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
cf_fig = sn.heatmap(df_cf_matrix, annot=True, fmt='d')
cf_fig = cf_fig.get_figure()
cf_fig.savefig("confusion_matrix.png", facecolor="white")
