import os
import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
#from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import roc_auc_score
import pydicom
from pydicom import dcmread
import random


class ChestXrayHDF5(Dataset):
    def __init__(self, path):
        hdf5_database = h5py.File(path, 'r')
        self.path = path
        self.hdf5_database = hdf5_database

    def __getitem__(self, index):
        hdf5_image = self.hdf5_database["img"][index, ...]  # read image
        image = torch.from_numpy(hdf5_image)
        # returns numpy.ndarray
        hdf5_label = self.hdf5_database["labels"][index, ...]
        label = [int(i) for i in hdf5_label]
        return image, torch.FloatTensor(label)

    def __len__(self):
        return self.hdf5_database["img"].shape[0]


class PydicomDataset(Dataset):
    def __init__(self, path):
        self.img_filenames = [os.path.join(path, x) for x in os.listdir(path)]

    def __getitem__(self, index, crop_size=64):
        dicom_img = dcmread(self.img_filenames[index]).pixel_array  # numpy array img
        #mean = 15397.0
        #std = 11227.0
        #img_normed = dicom_img / mean + std
        img_normed = dicom_img/(np.max(dicom_img)-np.min(dicom_img))
        img_torch = torch.from_numpy(img_normed)

        transform_ = transforms.RandomCrop(crop_size)
        img_cropped = transform_(img_torch)
        #return transforms.ToTensor()(img_cropped)
        return img_cropped.unsqueeze(0)

    def __len__(self):
        len_ = len(self.img_filenames)
        return len_

class PydicomDataset1D(Dataset):
    def __init__(self, path='/Users/mariaulyanova/Desktop/philips/data/Train_sins.npy', phase = 'train'):
        #self.img_filenames = [os.path.join(path, x) for x in os.listdir(path)]
        self.sinograms = np.load(path) #(2328, 1200, 512)
        self.sins = self.sinograms.reshape(self.sinograms.shape[0]*self.sinograms.shape[1],-1) #(2793600, 512)
        self.phase = phase
        self.to_tensor = ToTensor()

    def __getitem__(self, index, crop_size=64):
        if self.phase == 'train':
            ind = 0
        if self.phase == 'test':
            ind = 2790000
        sin = self.sins[index + ind, :]
        sin_torch = torch.squeeze(torch.from_numpy(sin.astype(np.float32))) # H x W
        return (sin_torch.unsqueeze(1).unsqueeze(0), index+ind)

    def __len__(self):
        if self.phase == 'train':
            return 2400000 #total 2793600
        if self.phase == 'test':
            return 60000  # max(len(self.sinograms),2000)

class PydicomDataset1DPatches(Dataset): #sample random 60 projections from one slice by index of slice
#samples 
    def __init__(self, path='/Users/mariaulyanova/Desktop/philips/data/Train_sins.npy', phase = 'train', n_projs = 60):
        #self.img_filenames = [os.path.join(path, x) for x in os.listdir(path)]
        self.sinograms = np.load(path) #(2328, 1200, 512)
        self.sins = self.sinograms.reshape(self.sinograms.shape[0]*self.sinograms.shape[1],-1) #(2793600, 512)
        self.phase = phase
        self.to_tensor = ToTensor()
        self.n_projs = n_projs

    def __getitem__(self, index, crop_size=64):
        if self.phase == 'train':
            ind = 0
        if self.phase == 'test':
            ind = 2640000
        patch = []
        if self.phase == 'train':
            slice_idx = index % 2200
        if self.phase == 'test':
            slice_idx = index%70
        inds = random.sample(range(1200), self.n_projs) #random projections
        for i in inds:
            sin = self.sins[slice_idx*1200 + i + ind, :]
            #print(sin[0], 'FFFFFFFF')
            sin = torch.squeeze(torch.from_numpy(sin.astype(np.float32)))
            #print(sin[0])
            sin = sin.unsqueeze(1).unsqueeze(0)
            #print(sin.shape)
            patch.append(sin)
        return torch.stack(patch), torch.stack([torch.tensor(i) for i in inds]), slice_idx
    def __len__(self):
        if self.phase == 'train':
            return 44000 #total 2793600 #if 44000 is amount of all samples of batches, and slice numbers are mod(2200) then we will get 20 samples from each slice
        if self.phase == 'test':
            return 70*20  # max(len(self.sinograms),2000)


class PydicomDatasetSeg(Dataset):
    def __init__(self, path='C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/Sinograms'):
        #self.img_filfenames = [os.path.join(path, x) for x in os.listdir(path)]
        self.names = [os.path.join(path,n) for n in sorted(os.listdir(path))] #names of sinograms in  npy


    def __getitem__(self, index, crop_size=64):
        path_i = self.names[31+index]
        arr = np.load(path_i, mmap_mode = 'r') #1000, 512
        #arr = (arr-157866.81)/147934.61 #std
        #arr = (arr+906573.3)/(624764.+906573.) #minmax
        arr = arr/607411.
        #arr = np.round(arr*8191.)/8192.
        arr = np.expand_dims(arr, axis = 2) #1000, 512, 1
        arr_torch = torch.from_numpy(arr)
        return arr_torch

    def __len__(self):
        return 70+99 #len(self.names)

class PydicomDatasetSegMask(Dataset):
    def __init__(self, path='C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/Sinograms', path_masks = 'C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/masks'):
        #self.img_filfenames = [os.path.join(path, x) for x in os.listdir(path)]
        self.names = [os.path.join(path,n) for n in sorted(os.listdir(path))] #names of sinograms in  npy
        self.names_mask = [os.path.join(path_masks,n) for n in sorted(os.listdir(path_masks))] #names of sinograms in  npy


    def __getitem__(self, index, crop_size=64):
        path_i = self.names[31+index]
        path_mask_i = self.names_mask[31+index]
        arr = np.load(path_i, mmap_mode = 'r') #1000, 512
        mask = np.load(path_mask_i, mmap_mode= 'r') #1, 512, 512
        #arr = (arr-157866.81)/147934.61 #std
        arr = arr/607411. #minmax
        arr = np.expand_dims(arr, axis = 2) #1000, 512, 1
        arr_torch = torch.from_numpy(arr)
        mask_torch = torch.from_numpy(mask)
        return arr_torch, mask_torch

    def __len__(self):
        return 70+99 #len(self.names)

class PydicomDatasetSegMaskQuantized(Dataset):
    def __init__(self, path='C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/Sinograms', path_masks = 'C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/masks'):
        #self.img_filfenames = [os.path.join(path, x) for x in os.listdir(path)]
        self.names = [os.path.join(path,n) for n in sorted(os.listdir(path))] #names of sinograms in  npy
        self.names_mask = [os.path.join(path_masks,n) for n in sorted(os.listdir(path_masks))] #names of sinograms in  npy


    def __getitem__(self, index, crop_size=64):
        path_i = self.names[31+index]
        path_mask_i = self.names_mask[31+index]
        arr = np.load(path_i, mmap_mode = 'r') #1000, 512
        mask = np.load(path_mask_i, mmap_mode= 'r') #1, 512, 512
        #arr = (arr-157866.81)/147934.61 #std
        arr = arr/607411. #minmax
        arr = np.round(arr*8192.)/8192.
        arr = np.expand_dims(arr, axis = 2) #1000, 512, 1
        arr_torch = torch.from_numpy(arr)
        mask_torch = torch.from_numpy(mask)
        return arr_torch, mask_torch

    def __len__(self):
        return 70+99 #len(self.names)

class CXRDataset(Dataset):
    def __init__(self, dataset, img_dir, image_list_file, img_size, num_label, view, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            mode: 'frontal' or 'lateral'
            transform: optional transform to be applied on a sample.
        """
        view = view.lower()
        image_names = []
        labels = []

        f = open(image_list_file, 'r')
        d = '\t'
        reader = csv.reader(f, delimiter=d)
        next(reader, None)
        for row in reader:
            if dataset == 'CheXpert':
                items = row[0].split(',')
                image_name = os.path.join(img_dir, items[0][20:])
                img_view = items[3].lower()
                label = items[5:]
            elif dataset == 'mimic':
                items = row[0].split(',')
                image_name = f'{img_dir}/{items[0]}'
                img_view = items[1].lower()
                label = items[2:]
            indices = [i for i, x in enumerate(label) if x == "1.0"]
            output = np.zeros(num_label)
            if img_view == view:
                for index in indices:
                    output[index] = 1
                    output[index] = int(output[index])
                labels.append(output)
                image_names.append(image_name)

        self.image_names = image_names
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its label
        """
        image_name = self.image_names[index]
        im = Image.open(image_name).convert('RGB')
        old_size = im.size
        ratio = float(self.img_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        # create pads
        image = Image.new("RGB", (self.img_size, self.img_size))
        # paste resized image in the middle of padding
        image.paste(im, ((self.img_size - new_size[0]) // 2,
                         (self.img_size - new_size[1]) // 2))
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


def recon_image(n_row, original_img, model, save_path, epoch, Tensor, experiment_comet = None):
    """Saves a grid of decoded / reconstructed digits."""
    model.eval()
    original_img = original_img[0:n_row ** 2, :]
    with torch.no_grad():
        out, _ = model(original_img)

    # remove normalization
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).type(Tensor)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).type(Tensor)
    original_img = original_img * std + mean
    out = out * std + mean

    save_image(original_img.data, f'{save_path}/sample/original.png',
               nrow=n_row, normalize=True, range=(0, 1))
    save_image(out.data, f'{save_path}/sample/{str(epoch + 1).zfill(4)}.png',
               nrow=n_row, normalize=True, range=(0, 1))
    save_image(torch.cat([original_img, out], 0).data, f'{save_path}/sample/flat_{str(epoch + 1).zfill(4)}.png',
               nrow=n_row**2, normalize=True, range=(0, 1))
    if experiment_comet is not None:
        experiment_comet.log_image(original_img.data, name=f'{save_path}/sample/original.png')
        experiment_comet.log_image(out.data, f'{save_path}/sample/{str(epoch + 1).zfill(4)}.png')
        experiment_comet.log_image(torch.cat([original_img, out], 0).data, f'{save_path}/sample/flat_{str(epoch + 1).zfill(4)}.png')
    model.train()



def recon_pydicom(n_row, original_img, model, save_path, epoch, Tensor):
    """Saves a grid of decoded / reconstructed digits."""
    model.eval()
    original_img = original_img[0:n_row ** 2, :] #only n**2 images from batch if n**2=25, batch 128 than number is 25
    #print('original img ', original_img.shape)
    with torch.no_grad():
        out, _ , _= model(original_img)
    #print('out', out.shape)
    # remove normalization
    sq_out=out #.squeeze(0) #[1, 64, 64] -> [64, 64]
    sq_original_img=original_img #.squeeze(0)
    mean = 15397.0
    std = 11227.0
    max = 65535.0

    img_orig = sq_original_img
    out_final = sq_out
    #print('')
    #print('AAAAAAAAAAAAA', img_orig.data.shape, torch.max(img_orig.data))
    #print('AAAAAAAAAAAAA', out_final.shape, torch.max(out_final))
    #print('bbbbbbbkxkjgkexjx', torch.cat([img_orig, out_final], dim=0).data.shape, torch.max(torch.cat([img_orig, out_final], dim=0)) )

    #out_final = out_final*255/(torch.max(out_final)-torch.min(out_final))
    #img_orig = img_orig*255/(torch.max(img_orig)-torch.min(img_orig))

    #print('BBBBBBBBBBBB', img_orig.data.shape, torch.max(img_orig.data))
    #print('BBBBBBBBBBBB', out_final.shape, torch.max(out_final))

    #img_orig = sq_original_img * std + mean
    #out_final = sq_out * std + mean

    #img_orig_255=img_orig*256/max
    #out_255=out_final*256/max

   #imwrite(rgb2gray(I1), f'{save_path}/sample/original', 'png');

    save_image(img_orig.permute((0,1,3,2)).data, f'{save_path}/sample/original.png',
               nrow=n_row, normalize=True)#, range=(0, 1))
    save_image(out_final.permute((0,1,3,2)).data, f'{save_path}/sample/{str(epoch + 1).zfill(4)}.png',
               nrow=n_row, normalize=True)#, range=(0, 1))
    save_image(torch.cat([img_orig.permute((0,1,3,2)).data, out_final.permute((0,1,3,2)).data], dim=0).data, f'{save_path}/sample/flat_{str(epoch + 1).zfill(4)}.png',
               nrow=n_row ** 2, normalize=True)#, range=(0, 1))
    model.train()


def compute_AUCs(gt, pred, N_CLASSES):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return np.asarray(AUROCs)


def save_loss_plots(n_epochs, latest_epoch, losses, save_path):
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, losses[0, :, 0], '-')
    ax1.plot(epochs, losses[1, :, 0], '-')
    ax1.set_title('All Losses')
    ax1.set_xlabel('Epochs')
    ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")

    plt.close(fig)
    fig.savefig(f'{save_path}/loss graphs.png')


def save_loss_plots_masha(n_epochs, latest_epoch, history, save_path, fold):
    fig = plt.figure(figsize=(12, 12),dpi = 200)
    latest_epoch += 1
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(history['train_loss'], marker = '.')
    ax1.plot(history['val_loss'], marker ='.')
    ax1.set_title('Compression total loss')
    ax1.set_xlabel('Epochs')
    #ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.close(fig)
    fig.savefig(f'{save_path}/loss_total_fold_{fold}.png')

    fig = plt.figure(figsize=(12, 12),dpi = 200)
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(history['train_loss_latent'], marker = '.')
    ax1.plot(history['val_loss_latent'], marker = '.')
    ax1.set_title('Latent MSE loss')
    ax1.set_xlabel('Epochs')
    #ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.close(fig)
    fig.savefig(f'{save_path}/loss_latent_fold_{fold}.png')

    fig = plt.figure(figsize=(12, 12),dpi = 200)
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(history['train_loss_recon'], marker = '.')
    ax1.plot(history['val_loss_recon'], marker = '.')
    ax1.set_title('MSE compression recon loss')
    ax1.set_xlabel('Epochs')
    #ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.close(fig)
    fig.savefig(f'{save_path}/loss_recon_mse_fold_{fold}.png')

    fig = plt.figure(figsize=(12, 12),dpi = 200)
    # BCE Loss
    ax1 = fig.add_subplot(111)
    ax1.plot(history['train_dice'], marker = '.')
    ax1.plot(history['val_dice'], marker = '.')
    ax1.set_title('DICE loss')
    ax1.set_xlabel('Epochs')
    #ax1.axis(xmin=1, xmax=latest_epoch)
    ax1.legend(["Train Loss", "Validation Loss"], loc="upper right")
    plt.close(fig)
    fig.savefig(f'{save_path}/dice_fold_{fold}.png')


def save_loss_AUROC_plots(n_epochs, latest_epoch, losses, AUROCs, save_path):
    epochs = range(1, n_epochs + 1)
    fig = plt.figure(figsize=(15, 15))
    latest_epoch += 1
    # BCE Loss
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, losses[0, :], '-')
    ax1.plot(epochs, losses[1, :], '-')
    ax1.legend(["Train", "Val"], loc="upper right")
    ax1.set_title('BCE Loss')
    ax1.set_xlabel('Epochs')
    ax1.axis(xmin=1, xmax=latest_epoch)

    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, np.mean(AUROCs[0, :], axis=1), '-')
    ax2.plot(epochs, np.mean(AUROCs[1, :], axis=1), '-')
    ax2.legend(["Train", "Val"], loc="upper right")
    ax2.set_title('Average AUROC')
    ax2.set_xlabel('Epochs')
    ax2.axis(xmin=0, xmax=latest_epoch)

    plt.close(fig)
    fig.savefig(f'{save_path}/loss and AUROC graphs.png')


def rgb2gray(original_img):
    r, g, b = original_img[0, :], original_img[1, :], original_img[2, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
