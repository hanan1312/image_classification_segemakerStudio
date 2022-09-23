import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import sys
import smdebug.pytorch as smd
import torch.nn.functional as F
import torch.utils.data.distributed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# logging_configs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, epoch_no, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    
    hook.set_mode(smd.modes.EVAL)
    
    running_loss = 0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += pred.eq(labels.view_as(pred)).sum().item()

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))

def train(model, train_loader, criterion, optimizer, device, hook, epoch_no):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset" )
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += pred.eq(labels.view_as(pred)).sum().item()
        running_samples += len(inputs)
        loss.backward()
        optimizer.step()
        if running_samples % 500 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*(running_corrects/ running_samples)
            ))
    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects/ len(train_loader.dataset)
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, running_corrects, len(train_loader.dataset), 100.0 * total_acc
    ))   
    return model
    
def net():
    model = models.resnet50(pretrained = True) #Use the pretrained resnet50 model with 50 layers
    
    for param in model.parameters():
        param.requires_grad = False #Freeze all the Conv layers
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 133),nn.ReLU(inplace = True))
    return model

def create_data_loaders(data, batch_size):
    
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dataset_path = os.path.join(data, "train")
    test_dataset_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor() ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )
    
    return train_data_loader, test_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    batch_size = args.batch_size
    data = args.data_dir
    path = args.model_dir
    lr= args.lr
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model=net()
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_data_loader, test_data_loader = create_data_loaders(data, batch_size )
      
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()

    hook.register_loss(loss_criterion)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    for epoch_no in range(1):
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model=train(model, train_data_loader, loss_criterion, optimizer, device, hook, epoch_no)
        test(model, test_data_loader, loss_criterion, device, epoch_no, hook)
        
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the Model ...")
    torch.save(model.state_dict(), os.path.join(path, 'my_model.pth'))
    logger.info("Model Saving completed  ...")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
                        

    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args=parser.parse_args()
    
    main(args)