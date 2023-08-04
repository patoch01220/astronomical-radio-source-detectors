import torch 
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os 
import myUtils
from astropy.io import fits
# from torchsummary import summary
from models.blobsfinder import BlobsFinder
from models.blobsfinder2 import BlobsFinder2
import yaml


# ------------------------------------------------------ #
parser = argparse.ArgumentParser(description='...')
parser.add_argument("--data_path", default='MasterThesis/Dataset.nosync', type=str, help="....")
parser.add_argument('--seed', default=42,  type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size', default=64,  type=int, help='...')
parser.add_argument('--lr', default=0.01,  type=float, help='...')
parser.add_argument('--epochs', default=400,  type=int, help='')
parser.add_argument('--weight_decay', default=0,  type=float, help='...')
parser.add_argument('--hidden_channels', default=256,  type=int, help='')

parser.add_argument('--is_stochastic', default=True,  type=int, help='...')
parser.add_argument('--noise_std', default=0.2,  type=float, help='...')

parser.add_argument("--save_path", default='./Results/plots', type=str, help="....")
# ------------------------------------------------------ #

# def trainingModel(input, target, device, criterion, epochs, batch_size, hidden_channels, lr_list, weight_decay=0, modelName='modeltmp', Training = True):
def trainingModel(dataloader, device, criterion, args, Training=True, modelName=""):

    # model = BlobsFinder(args.hidden_channels)
    model = BlobsFinder2(args.hidden_channels)
    # summary(model, (1, 512, 512))
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if Training:
        print("Training..")
        
        model.to(device)
        # input = input.to(device)
        # target = target.to(device)

        lossHistory = []
        for e in range(args.epochs):
            # if e == 50:
            #     lr = args.lr/10
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

            acc_loss = 0
            for step, (input, target) in enumerate(dataloader): 
                output = model(input.unsqueeze(1).to(device))
                if args.is_stochastic:
                    target += torch.normal(torch.zeros(target.shape), torch.ones(target.shape)*args.noise_std)

                loss = criterion(output.squeeze(1), target.to(device))     

                acc_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lossHistory.append(acc_loss)
            print(f"\tEpoch: {e+1}, loss ={acc_loss}")

            if e%20 == 0:
                fig = plt.figure()
                dr = 1; dc = 3; il = 0
                il += 1; plt.subplot(dr, dc, il)
                plt.imshow(input[0].detach().cpu().numpy())
                plt.axis("off")
                plt.title("input")

                il += 1; plt.subplot(dr, dc, il)
                plt.imshow(target[0].detach().cpu().numpy())
                plt.axis("off")
                plt.title("target")

                il += 1; plt.subplot(dr, dc, il)
                plt.imshow(output.squeeze(1)[0].detach().cpu().numpy())
                plt.axis("off")
                plt.title("output")

                plt.tight_layout()
                plt.savefig(os.path.join(args.save_path, "epoch_%s.pdf" % (e+1))) # _noisy
                plt.close(fig)

            if 200 < e < 301:
                if e % 20 == 0:
                    torch.save({
                        'epoch': args.epochs,
                        'batch_size': args.batch_size,
                        'hidden_channels': args.hidden_channels,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss_history': lossHistory,
                        }, './trainedModels/noisy/'+ modelName+'_final_e'+str(e)+'.pt'
                    )
            elif e > 300:
                if e % 50 == 0:
                    torch.save({
                        'epoch': args.epochs,
                        'batch_size': args.batch_size,
                        'hidden_channels': args.hidden_channels,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss_history': lossHistory,
                        }, './trainedModels/noisy/'+ modelName+'_final_e'+str(e)+'.pt'
                    )
        # --- Saving the model --- #     
        torch.save({
            'epoch': args.epochs,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': lossHistory,
            }, './trainedModels/noisy/'+ modelName+'_final_e'+str(e)+'.pt'
        )
        return model, lossHistory
    else:
        print("Testing...")

        loadedModel = torch.load('./trainedModels/'+modelName+'_final'+'.pt')
        epochs, batch_size, hidden_channels = loadedModel['epoch'], loadedModel['batch_size'], loadedModel['hidden_channels']
        model.load_state_dict(loadedModel['model_state_dict'])
        optimizer.load_state_dict(loadedModel['optimizer_state_dict'])
        lossHistory = loadedModel['train_loss_history']
    
        return model, optimizer, epochs, batch_size, hidden_channels, lossHistory

def makeDir(dir):
    """Verifies if a directory exists and creates it if does not exist

    Args:
        dir: directory path

    Returns:
        str: directory path """

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir
# ------------------------------------------------------ #
if __name__ == "__main__":

    # Detect device
    print("Cuda available: ", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    args = parser.parse_args()
    print(yaml.dump(args, allow_unicode=True, default_flow_style=False))

    home = str(Path.home())
    data_path = os.path.join(home, args.data_path)
    makeDir(args.save_path)

    # --- Loading the data --- #
    keys = np.load(os.path.join(data_path, "Sources_detection/sky_keys.npy"))
    indices_train = np.load(os.path.join(data_path, "Sources_detection/seed_0/train_idx.npy"))
    indices_validation = np.load(os.path.join(data_path, "Sources_detection/seed_0/validation_idx.npy"))
    indices_test = np.load(os.path.join(data_path, "Sources_detection/seed_0/test_idx.npy"))

    print("---"*15)
    print("Numbber of keys: ", len(keys))
    print("Lenght train set: ", len(indices_train))
    print("Lenght validation set: ", len(indices_validation))
    print("Lenght test set: ", len(indices_test))
    print("---"*15)
    print()


    # List of images paths regarding the train, validation and test set
    train_images_path = [os.path.join(data_path, 'clean_noisy', "clean_noisy_gaussians_" + keys[ind] + ".fits") for ind in indices_train]
    validation_images_path = [os.path.join(data_path, 'clean_noisy', "clean_noisy_gaussians_" + keys[ind] + ".fits") for ind in indices_validation]
    test_images_path = [os.path.join(data_path, 'clean_noisy', "clean_noisy_gaussians_" + keys[ind] + ".fits") for ind in indices_test]

    print("---"*15)
    print("Lenght train set: ", len(train_images_path))
    print("Lenght validation set: ", len(validation_images_path))
    print("Lenght test set: ", len(test_images_path))
    print("---"*15)
    print()

    train_RealDetection = myUtils.loadAllCatFile(data_path + "/Sources_detection/cat", "gaussians_", indices_train, keys)
    validation_RealDetection = myUtils.loadAllCatFile(data_path + "/Sources_detection/cat", "gaussians_", indices_validation, keys)
    test_RealDetection = myUtils.loadAllCatFile(data_path + "/Sources_detection/cat", "gaussians_", indices_test, keys)

    print("---"*15)
    print("Lenght train set: ", len(train_RealDetection))
    print("Lenght validation set: ", len(validation_RealDetection))
    print("Lenght test set: ", len(test_RealDetection))
    print("---"*15)
    print()

    # compute the pixels index of ground truth positions
    train_RealDetection_px = [myUtils.RaDec2pixels(train_RealDetection[ind], fits.getheader(train_images_path[ind])) for ind in range(len(train_RealDetection))]
    validation_RealDetection_px = [myUtils.RaDec2pixels(validation_RealDetection[ind], fits.getheader(validation_images_path[ind])) for ind in range(len(validation_RealDetection))]
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
    validation_input = [fits.getdata(validation_images_path[ind])[0,0] for ind in range(len(validation_images_path))]
    validation_input = torch.nan_to_num(torch.tensor(np.array(validation_input)), nan=0)

    # open fits files from test_images_path and store them into a tensor test_input
    test_input = [fits.getdata(test_images_path[ind])[0,0] for ind in range(len(test_images_path))]
    test_input = torch.nan_to_num(torch.tensor(np.array(test_input)), nan=0)
    print("---"*15)
    print("shape train input: ", train_input.shape)
    print("shape validation input: ", validation_input.shape)
    print("shape test input: ", test_input.shape)
    print("---"*15)


    # ## Creating the target dataset
    # The target dataset is a zero image with 1 where there is a source. To create this tensor we use the list of real position of the source and set the pixels at this location to 1.

    binaryMaps = np.load(os.path.join(data_path, "Sources_detection/sky_true_local_norm.npy"))
    print("Binary map shape: ",binaryMaps.shape)

    train_binaryMaps = torch.tensor(binaryMaps[indices_train])
    validation_binaryMaps = torch.tensor(binaryMaps[indices_validation])
    test_binaryMaps = torch.tensor(binaryMaps[indices_test])

    print("Lenght train binMap: ", train_binaryMaps.shape)
    print("Lenght validation binMap: ", validation_binaryMaps.shape)
    print("Lenght test binMap: ", test_binaryMaps.shape)

    # Note that I had to change some dimension size in the Deep focus source code: they had as input image of 256x256 and in our case, they are of size 512x512. So I had to change the "16" values at line 70, 93 and 96 in the blobfinder code with the value "32" which suits the desired size.

    # epochs = 500
    # batch_size = 64
    # hidden_channels = 64
    # weight_decay = 0
    # lr = [0.01,0.001]
    criterion = nn.L1Loss()

    # muli_gpu = False
    # mode = 'train'
    # weight_decay = 1e-5
    # weight_decay = 0
    # early_stopping = 'True'
    # patience = 20
    # warm_start=True
    # warm_start_iterations = 10
    # detection_threshold = 0.15

    dataloader = DataLoader(TensorDataset(train_input, train_binaryMaps), batch_size=args.batch_size, shuffle=True) # create your dataloader
    print()
    print("batch size: ", args.batch_size)
    print("hidden channels: ", args.hidden_channels)
    print("epochs: ", args.epochs)

    print("Training..")
    model, lossHistory = trainingModel(dataloader, device, criterion, args=args, modelName='model_test',)

    # ----------------------------------- #
    print("Validation..")
    print("Loading validation data..")
    # test_input = test_input.to(device)
    # test_binaryMaps = test_binaryMaps.to(device)

    validation_input = validation_input.to(device)
    validation_binaryMaps = validation_binaryMaps.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(validation_input.unsqueeze(1))
        print("Validation loss: ", criterion(pred.squeeze(1), validation_binaryMaps).item(), "\n")
