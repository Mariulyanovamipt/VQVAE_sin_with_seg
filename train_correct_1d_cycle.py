import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from networks import VQVAE, VQVAE1D
from utilities import recon_pydicom, save_loss_plots, save_loss_plots_masha, PydicomDataset1DPatches, PydicomDatasetSeg, PydicomDatasetSegMask, PydicomDatasetSegMaskQuantized
import random
#from comet_ml import Experiment
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12.0, 12.0]
plt.rcParams['figure.dpi'] = 200

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from Unet_model import UNet
from piq import psnr, ssim
from modules import VectorQuantizedVAE, VQVAEMonster1D
#experiment_comet = Experiment(api_key="SCNv0oqcoLPJnXIqei3EvhuKR", project_name=f"VQVAE_srv_recon_loss",workspace='mariulyanovamipt')

#from torch_radon import Radon, RadonFanbeam

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--first_stride', type=int, default=16, help="2, 4, 8, or 16")
parser.add_argument('--second_stride', type=int, default=8, help="2, 4, 8, or 16")
parser.add_argument('--embed_dim', type=int, default=8)
#parser.add_argument('--data_path', type=str, default='/home/aisinai/work/HDF5_datasets')
parser.add_argument('--data_path', type=str, default='C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/Sinograms')
parser.add_argument('--data_path_masks', type=str, default='C:/Users/User/Desktop/philips/data/C002_toy_cvat_coco/masks')
#MY
#parser.add_argument('--test_data_path', type=str, default='/home/m-ulyanova/data/Train_sins.npy')
#parser.add_argument('--dataset', type=str, default='CheXpert', help="CheXpert or mimic")
parser.add_argument('--dataset', type=str, default='MY', help="CheXpert or mimic")
#parser.add_argument('--view', type=str, default='frontal', help="frontal or lateral")
#parser.add_argument('--save_path', type=str, default='/home/aisinai/work/VQ-VAE2/20200820/vq_vae')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--train_run', type=str, default='monster_1D_one_by_one_crop_128_recon_1_1_0.5_bs_4')
args = parser.parse_args()
torch.manual_seed(816)

args = parser.parse_args()
torch.manual_seed(816)

#########################################################################
def random_crop_masha(batch, size_ = 512): #bs, 1, 1000, 512
    stack = []
    for img in batch:
        x = np.random.randint(0, high =1000-size_+1)
        y = np.random.randint(0, high = 512-size_+1)
        stack.append(img[:,x:x+size_,y:y+size_])
    return torch.stack(stack)


#########################################################################
save_path = f'{args.save_path}/{args.train_run}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/checkpoint/', exist_ok=True)
os.makedirs(f'{save_path}/sample/', exist_ok=True)
os.makedirs(f'{save_path}/images/',exist_ok = True)
with open(f'{save_path}/args.txt', 'w') as f:
    for key in vars(args).keys():
        f.write(f'{key}: {vars(args)[key]}\n')
        print(f'{key}: {vars(args)[key]}')
#fanbeam_angles = np.linspace(0, 2*np.pi, 1000, endpoint = False)
#radon_fanbeam = RadonFanbeam(512, fanbeam_angles, source_distance = 800, det_distance = 800, det_spacing = 2.5)
######################################################################
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        #outputs = outputs.squeeze()
        #targets = targets.float()
        #outputs = outputs.float()

        smooth = torch.tensor(1e-15).float()
        target = (targets > 0.0).float()
        prediction = torch.sigmoid(outputs) #.float()
        dice = (1 - (2 * torch.sum(prediction * target, dim=(2,3)) + smooth) / \
                     (torch.sum(prediction, dim=(2,3)) + torch.sum(target, dim=(2,3)) + smooth))
        return dice.mean()

class ReconDice(nn.Module):
    def __init__(self):
        super(ReconDice, self).__init__()
        #self.fanbeam_angles = np.linspace(0, 2*np.pi, 1000, endpoint = False)
        #self.radon_fanbeam = RadonFanbeam(512, self.fanbeam_angles, source_distance = 800, det_distance = 800, det_spacing = 2.5)
        model_seg = UNet(n_classes=1)
        model_seg.load_state_dict(torch.load("./tt_1_0.99010.pt")['state_dict'])
        self.device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")
        model_seg = model_seg.to(self.device)
        for p in model_seg.parameters():
            p.requires_grad_(False)
        self.seg_model = model_seg.eval()

    def forward(self, ideal_mask, compressed_sin, save_imgs = None): # 1000 x4 x 1 x512 x 1
        compressed_sin = torch.squeeze(compressed_sin, axis = 1) #1000,4,512
        compressed_sin_unnormalized = compressed_sin*607411. #(624764.+906573.)-906573.3
        #print('qq1')
        compressed_sin_filtered = radon_fanbeam.filter_sinogram(compressed_sin_unnormalized, 'ram-lak')
        #print('qq2 filtered', compressed_sin_filtered.shape)
        recon_compressed = radon_fanbeam.backprojection(compressed_sin_filtered)
        #print('qq 3 backproj')
        #print(compressed_sin_filtered.shape, recon_compressed.shape)
        recon_compressed =  torch.unsqueeze(recon_compressed, axis = 1)
        recon_compressed = ((recon_compressed.clip(-30, 1900) + 30) / 1900) #/255. #Yaroslav
        seg_mask_compressed = self.seg_model(recon_compressed)
        #print('qq5 after segment')
        loss_dice = DiceLoss()
        return 1-loss_dice(seg_mask_compressed.to(self.device), ideal_mask.to(self.device))
###############################################################
def save_recons(slice_orig, slice_compressed, mask, epoch, fold):
        compressed_sin_unnormalized = slice_compressed*607411. #*(624764.+906573.)-906573.3
        sin_unnormalized = slice_orig*607411. #(624764.+906573.)-906573.3 #print('qq1')
        compressed_sin_filtered = radon_fanbeam.filter_sinogram(compressed_sin_unnormalized, 'ram-lak')
        sin_filtered = radon_fanbeam.filter_sinogram(sin_unnormalized, 'ram-lak')
        #print('qq2 filtered', compressed_sin_filtered.shape)
        recon = radon_fanbeam.backprojection(sin_filtered)
        recon_compressed = radon_fanbeam.backprojection(compressed_sin_filtered)
        for i,img in enumerate(recon.squeeze()):
            plt.imsave(f'{save_path}/images/orig_slice_{i}_fold_{fold}.png',img.cpu().numpy())
        for i,img in enumerate(recon_compressed.squeeze()):
            plt.imsave(f'{save_path}/images/compr_slice_{i}_epoch_{epoch}_fold_{fold}.png',img.cpu().numpy())
        plt.imsave(f'{save_path}/images/mask_{i}_fold_{fold}.png',mask.cpu().numpy())
##########################################################

k=5
batch_size = 4
splits=KFold(n_splits=k,shuffle=True,random_state=42)

full_dataset = PydicomDatasetSegMaskQuantized(path=f'{args.data_path}', path_masks=f'{args.data_path_masks}')
L = len(full_dataset)
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(full_dataset)))):
#for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(50))):
    print('Fold {}'.format(fold + 1))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    DS, dataloaders = {}, {}
    #DS['train'], DS['test'] = train_dataset, test_dataset
    dataloaders['train'] = train_loader
    dataloaders['valid'] = test_loader
    for i,smpl in enumerate(dataloaders['valid']):
        sample_patch = Variable(smpl[0].permute((0,3,1,2))) #.unsqueeze(1) #.type(Tensor) #b x 1000 x h x w -> b,1, h, w
        break

    ####### model #######
    if cuda:
        model = VQVAEMonster1D(in_channel=1, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=32, decay=0.99, #n_embed default 512
                 first_stride=4, second_stride=4).cuda()
    else:
        model = VQVAEMonster1D(in_channel=1, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=32, decay=0.99, #n_embed default 512
                 first_stride=4, second_stride=4)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        device_ids = list(range(n_gpu))
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay= 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)
    ####### model ######
    losses = np.zeros((2, args.n_epochs, 3))  # [0,:,:] index for train, [1,:,:] index for valid

    history = {'train_loss': [], 'val_loss': [],
               'train_loss_latent':[], 'val_loss_latent':[],
               'train_loss_recon':[],'val_loss_recon':[],
               'train_dice':[], 'val_dice':[],
               'train_psnr':[], 'val_psnr':[],
               'train_ssim':[], 'val_ssim':[]
               }

    for epoch in range(args.n_epochs):
        cur_loss_train, cur_loss_val = [], []
        cur_loss_lat_train, cur_loss_lat_val = [], []
        cur_loss_recon_train, cur_loss_recon_val = [], []
        cur_loss_dice_train, cur_loss_dice_val = [], []
        cur_psnr_img_val, cur_psnr_val = [], []
        cur_ssim_img_val, cur_ssim_val = [], []
        for phase in ['train', 'valid']:
            model.train(phase == 'train')  # True when 'train', False when 'valid'
            criterion = nn.MSELoss()
            n_row = 5
            loader = tqdm(dataloaders[phase])
            for i, (slice_, mask) in enumerate(loader): #slice bs,1000,512,1
                slice_ = slice_.permute((0,3,1,2)) #bs 1 1000 512
                mask = Variable(mask.type(Tensor))
                slice_ = Variable(slice_.type(Tensor))
                slice_crop = Variable(random_crop_masha(slice_).type(Tensor))

                inds = random.sample(range(1000), 500) #500 random unrepeated indx
                for j in range(50):
                    patch = slice_[:,:,inds[j],:] # -> Bs x h x w
                    patch = torch.unsqueeze(patch, dim=1) #B x ch x H x W
                    patch = Variable(patch.type(Tensor))
                    with torch.set_grad_enabled(phase == 'train'):
                            optimizer.zero_grad()
                            #print('\n pathch statistics: ', slice_crop.min(), slice_crop.max(), slice_crop.mean(), slice_crop.std())
                            #print(slice_crop.shape)
                            out_proj ,qq_t,qq_b = model(patch)
                            print('---------', out_proj.shape, patch.shape)
                            #print('out statistics ', out_crop.min(), out_crop.max(), out_crop.mean(), out_crop.std())

                            loss_recons = F.mse_loss(out_proj,patch)
                            # Vector quantization objective
                            loss_vq = F.mse_loss(qq_t[0], qq_t[1].detach()) + F.mse_loss(qq_b[0], qq_b[1].detach())
                            # Commitment objective
                            loss_commit = F.mse_loss(qq_t[1], qq_t[0].detach())+ F.mse_loss(qq_b[1], qq_b[0].detach())

                            loss = 1.*loss_recons + 1.*loss_vq + 0.5* loss_commit
                            #segmentation_loss = ReconDice()(mask, out)
                            #loss = loss_vae +  0.25*segmantation_loss
                            if phase == 'train':
                                optimizer.zero_grad()
                                #with torch.no_grad():
                                    #seg_loss = ReconDice()(mask,out)
                                loss.backward()
                                optimizer.step()

                                #logging
                                losses[0, epoch, :] = [loss.cpu().detach().numpy(), loss_recons.cpu().detach().numpy(), loss_vq.cpu().detach().numpy() ]
                                cur_loss_train.append(loss.cpu().detach().numpy().mean())
                                #cur_loss_recon_train.append(recon_loss.cpu().detach().numpy().mean())
                                #cur_loss_lat_train.append( latent_loss.cpu().detach().numpy().mean())
                                #cur_loss_dice_train.append(seg_loss.cpu().detach().numpy().mean())
                                #cur_psnr_train.append(psnr(slice_,out_).cpu().detach().numpy().mean())
                                #cur_ssim_train.append(ssim(slice_out_).cpu().detach().numpy().mean())

                                scheduler.step()
                                lr = optimizer.param_groups[0]['lr']
                                #scheduler.step()
                            else:
                                #if i == 1:
                                    #save_recons(slice_, out, mask, epoch, fold)
                                    #seg_loss = ReconDice()(mask,out, save_imgs=(i,fold, epoch))

                                #seg_loss = ReconDice()(mask,out)
                                #loss = loss_vae # + 0.25*seg_loss

                                losses[1, epoch, :] = [loss.cpu().detach().numpy(), loss_recons.cpu().detach().numpy(), loss_vq.cpu().detach().numpy()]
                                cur_loss_val.append(loss.cpu().detach().numpy().mean())
                                #cur_loss_recon_val.append(recon_loss.cpu().detach().numpy().mean())
                                #cur_loss_lat_val.append( latent_loss.cpu().detach().numpy().mean())
                                #cur_loss_dice_val.append(seg_loss.cpu().detach().numpy().mean())
                                #cur_psnr_val.append(psnr(slice_,torch.clamp(out,0.,1.)).cpu().detach().numpy().mean())
                                #cur_ssim_val.append(ssim(slice_,torch.clamp(out,0.,1.)).cpu().detach().numpy().mean())
                                lr = optimizer.param_groups[0]['lr']
                                #scheduler.step()
                    loader.set_description((f'phase: {phase}; epoch: {epoch + 1}; total_loss: {loss.item():.5f}; '
                                            f'latent: {loss_recons.item():.5f}; mse: {loss_vq.item():.5f}; dice:{loss_commit.item()}'
                                            f'lr: {lr:.5f}'))
                    if i % 100 == 0:
                            #print('\n SHAPE SLICE', sample_patch.shape) #4 1 1000 512
                            recon_pydicom(n_row, torch.unsqueeze(sample_patch[:,:,100,:],dim=1), model, f'{save_path}', epoch, Tensor)
                            slice_compressed = []
                            for j in range(1000):
                                #print('\n SHAPE SLICE', sample_patch.shape)
                                patch = slice_[:,:,j,:] # -> Bs x h x w
                                patch = torch.unsqueeze(patch, dim=1) #B x ch x H x W
                                patch = Variable(patch.type(Tensor))
                                print(patch.shape)
                                out_proj ,_,_ = model(patch)
                                print(out_proj.shape)
                                slice_compressed.append(out_proj)
                            slice_compressed = torch.squeeze(torch.stack(slice_compressed))
                            print(slice_compressed.shape)
                            plt.imsave(f'{save_path}/sample/qq{epoch}_{i}.png', slice_compressed.detach().numpy()[:,2,:])

        history['train_loss'].append(np.mean(cur_loss_train))
        history['val_loss'].append(np.mean(cur_loss_val))

        history['train_loss_latent'].append(np.mean(cur_loss_lat_train))
        history['val_loss_latent'].append(np.mean(cur_loss_lat_val))

        history['train_loss_recon'].append(np.mean(cur_loss_recon_train))
        history['val_loss_recon'].append(np.mean(cur_loss_recon_val))

        history['train_dice'].append(np.mean(cur_loss_dice_train))
        history['val_dice'].append(np.mean(cur_loss_dice_val))
        history['val_psnr'].append(np.mean(cur_psnr_val))
        history['val_ssim'].append(np.mean(cur_ssim_val))

        save_loss_plots(args.n_epochs, epoch, losses, f'{save_path}')
        save_loss_plots_masha(args.n_epochs, epoch, history, f'{save_path}', fold)
        if (epoch) % 2 == 0:
            torch.save(model.state_dict(), f'{save_path}/checkpoint/vqvae2_1d_{str(epoch + 1).zfill(3)}_fold_{fold}.pt')
            np.save(f'{save_path}/vqvae2_1d_{str(epoch + 1).zfill(3)}_fold_{fold}.npy', history)
