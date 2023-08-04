# import utils.model_utils as mut
import torch 
from torch import nn
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt 
import os 
# import wandb
import myUtils
from tqdm import tqdm
from astropy.io import fits
from models.blobsfinder import BlobsFinder
from models.deepgru import DeepGRU
from models.resnet import ResNet18
from models.resnet import ResBlock

# # As usual: loading the data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

data_path = './Dataset.nosync'
keys = np.load(os.path.join(data_path, "Sources_detection/sky_keys.npy"))
indices_train = np.load(os.path.join(data_path, "Sources_detection/seed_0/train_idx.npy"))
# indices_validation = np.load(os.path.join(data_path, "Sources_detection/seed_0/validation_idx.npy"))
indices_test = np.load(os.path.join(data_path, "Sources_detection/seed_0/test_idx.npy"))

print("---"*15)
print("Numbber of keys: ", len(keys))
print("Lenght train set: ", len(indices_train))
# print("Lenght validation set: ", len(indices_validation))
print("Lenght test set: ", len(indices_test))
print("---"*15)
print()


# List of images paths regarding the train, validation and test set
train_images_path = [os.path.join(data_path, 'clean_gaussian', "clean_gaussians_" + keys[ind] + ".fits") for ind in indices_train]
# validation_images_path = [os.path.join(data_path, 'clean_gaussian', "clean_gaussians_" + keys[ind] + ".fits") for ind in indices_validation]
test_images_path = [os.path.join(data_path, 'clean_gaussian', "clean_gaussians_" + keys[ind] + ".fits") for ind in indices_test]

print("---"*15)
print("Lenght train set: ", len(train_images_path))
# print("Lenght validation set: ", len(validation_images_path))
print("Lenght test set: ", len(test_images_path))
print("---"*15)
print()

train_RealDetection = myUtils.loadAllCatFile(data_path+"/Sources_detection/cat", "gaussians_", indices_train, keys)
# validation_RealDetection = myUtils.loadAllCatFile("./Dataset.nosync/Sources_detection/cat", "gaussians_", indices_validation, keys)
test_RealDetection = myUtils.loadAllCatFile(data_path+"/Sources_detection/cat", "gaussians_", indices_test, keys)

print("---"*15)
print("Lenght train set: ", len(train_RealDetection))
# print("Lenght validation set: ", len(validation_RealDetection))
print("Lenght test set: ", len(test_RealDetection))
print("---"*15)
print()

# compute the pixels index of ground truth positions
train_RealDetection_px = [myUtils.RaDec2pixels(train_RealDetection[ind], fits.getheader(train_images_path[ind])) for ind in range(len(train_RealDetection))]
# validation_RealDetection_px = [myUtils.RaDec2pixels(validation_RealDetection[ind], fits.getheader(validation_images_path[ind])) for ind in range(len(validation_RealDetection))]
test_RealDetection_px = [myUtils.RaDec2pixels(test_RealDetection[ind], fits.getheader(test_images_path[ind])) for ind in range(len(test_RealDetection))]

print("---"*15)
print(torch.tensor(test_RealDetection_px[0]).shape)
print(torch.tensor(test_RealDetection_px[1]).shape)
print(torch.tensor(test_RealDetection_px[2]).shape)
print("---"*15)
print()

# ## Creatiung the training input:
# An input of our nn is simply the image contained in the fits file

# open fits files from train_images_path and store them into a tensor train_input
train_input = [fits.getdata(train_images_path[ind], memmap = False)[0,0] for ind in range(len(train_images_path))]
train_input = torch.nan_to_num(torch.tensor(np.array(train_input)), nan=0)

# open fits files from validation_images_path and store them into a tensor validation_input
# validation_input = [fits.getdata(validation_images_path[ind])[0,0] for ind in range(len(validation_images_path))]
# validation_input = torch.nan_to_num(torch.tensor(np.array(validation_input)), nan=0)

# open fits files from test_images_path and store them into a tensor test_input
test_input = [fits.getdata(test_images_path[ind])[0,0] for ind in range(len(test_images_path))]
test_input = torch.nan_to_num(torch.tensor(np.array(test_input)), nan=0)
print("---"*15)
print("shape train input: ", train_input.shape)
# print("shape validation input: ", validation_input.shape)
print("shape test input: ", test_input.shape)
print("---"*15)


# ## Creating the target dataset
# The target dataset is a zero image with 1 where there is a source. To create this tensor we use the list of real position of the source and set the pixels at this location to 1.

binaryMaps = np.load(os.path.join(data_path, "Sources_detection/sky_true_local_norm.npy"))
print("Binary map shape: ",binaryMaps.shape)

train_binaryMaps = torch.tensor(binaryMaps[indices_train])
# validation_binaryMaps = torch.tensor(binaryMaps[indices_validation])
test_binaryMaps = torch.tensor(binaryMaps[indices_test])

print("Lenght train binMap: ", train_binaryMaps.shape)
# print("Lenght validation binMap: ", validation_binaryMaps.shape)
print("Lenght test binMap: ", test_binaryMaps.shape)

# Note that I had to change some dimension size in the Deep focus source code: they had as input image of 256x256 and in our case, they are of size 512x512. So I had to change the "16" values at line 70, 93 and 96 in the blobfinder code with the value "32" which suits the desired size.
# 


def trainingModel(input, target, device, criterion, epochs, batch_size, hidden_channels, lr_list, weight_decay=0, modelName='modeltmp', Training = True):
    lr_index = 0
    lr = lr_list[lr_index]

    model = ResBlock(1, 1, False)
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay)

    if Training:
        print("Training..")
        
        model.to(device)
        input = input.to(device)
        target = target.to(device)

        lossHistory = []
        for e in range(epochs):
            if e == 250:
                lr_index += 1
                lr = lr_list[lr_index]
                for g in optimizer.param_groups:
                    g['lr'] = lr
            # if e == 50:
            #     lr_index += 1
            #     lr = lr_list[lr_index]
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

            acc_loss = 0
            for b in range(0, input.size(0), batch_size):    # using some mini batch 
                if b+batch_size <= input.shape[0]:                 # If the remaining dataset is large enough for a minibatch
                    output = model(input.narrow(0, b, batch_size).unsqueeze(1))   # unsqueeze because we need data to have shape 64x1x512x512 and not 64x512x512
                    
                    # tmp = torch.normal(torch.zeros(target.narrow(0, b, batch_size).shape), torch.ones(target.narrow(0, b, batch_size).shape)*0.2).to(device)
                    # loss = criterion(output.squeeze(1), target.narrow(0, b, batch_size)+tmp)
                    loss = criterion(output.squeeze(1), target.narrow(0, b, batch_size))
                else:               # Else we train on the remianing inputs
                    output = model(input.narrow(0, b, input.shape[0]-b).unsqueeze(1))   # unsqueeze because we need data to have shape 64x1x512x512 and not 64x512x512
                    # tmp = torch.normal(torch.zeros(target.narrow(0, b, target.shape[0]-b).shape), torch.ones(target.narrow(0, b, target.shape[0]-b).shape)*0.2).to(device)
                    # loss = criterion(output.squeeze(1), target.narrow(0, b, target.shape[0]-b)+tmp)
                    loss = criterion(output.squeeze(1), target.narrow(0, b, target.shape[0]-b))
                
                acc_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lossHistory.append(acc_loss)
            print(f"\tEpoch: {e+1}, loss ={acc_loss}")

        # --- Saving the model --- #     
        torch.save({
            'epoch': epochs,
            'batch_size': batch_size,
            'hidden_channels': hidden_channels,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': lossHistory,
            }, './trainedModels/'+modelName+'_resnet'+'.pt'
        )
        return model, optimizer, epochs, batch_size, hidden_channels, lossHistory
    else:
        print("Testing...")

        loadedModel = torch.load('./trainedModels/'+modelName+'_resnet'+'.pt')
        epochs, batch_size, hidden_channels = loadedModel['epoch'], loadedModel['batch_size'], loadedModel['hidden_channels']
        model.load_state_dict(loadedModel['model_state_dict'])
        optimizer.load_state_dict(loadedModel['optimizer_state_dict'])
        lossHistory = loadedModel['train_loss_history']
    
        return model, optimizer, epochs, batch_size, hidden_channels, lossHistory


epochs = 500
batch_size = 64
muli_gpu = False
mode = 'train'
# learning_rate = 0.001
# learning_rate = 0.01
# weight_decay = 1e-5
weight_decay = 0
early_stopping = 'True'
patience = 20
# hidden_channels = 1024
# hidden_channels = 512
hidden_channels = 512
warm_start=True
warm_start_iterations = 10
detection_threshold = 0.15


torch.manual_seed(42)


print("---"*15)
print()
print("Training..")

lr = [0.01, 0.001]
# criterion = nn.MSELoss()
criterion = nn.L1Loss()

# train_binaryMaps = train_binaryMaps + np.random.normal(0,0.2,train_binaryMaps.shape)
model, optimizer, epochs, batch_size, hidden_channels, lossHistory = trainingModel(train_input, train_binaryMaps, device, criterion, 
                                                    epochs, batch_size, hidden_channels, lr, weight_decay=weight_decay, 
                                                    modelName='model_test', Training=True)
# print()
model.eval()

print("Validation..")

print("Loading validation data..")
test_input = test_input.to(device)
test_binaryMaps = test_binaryMaps.to(device)

with torch.no_grad():
    pred = model(test_input.unsqueeze(1))
    print("Validation loss: ", criterion(pred.squeeze(1), test_binaryMaps).item())
    print()