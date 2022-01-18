import argparse
import random
import torch
import numpy as np
import os
import pandas as pd
from monai.transforms import (
    Compose, Resized, ToTensord, NormalizeIntensityd, 
    apply_transform, CenterSpatialCropd, RandScaleIntensityd,
    ScaleIntensityd, RandSpatialCropd, RandFlipd)
from pylab import rcParams
import matplotlib
import matplotlib.pyplot as plt
if "RTX 20" not in torch.cuda.get_device_name(0): matplotlib.use('Agg')
from monai.networks.nets import UNet, SegResNet, UNETR
from monai.losses import DiceLoss
from tqdm import tqdm

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    
class MandibleData(torch.utils.data.Dataset):
    def __init__(self, csv, args, mode, get_original=False):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.get_original = get_original
        self.args = args
    def __len__(self):
        return len(self.csv)
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        img_dir = os.path.join(self.args.input_path, "npy", f"{row.ID}_image.npy").replace(" ","")
        mask_dir = os.path.join(self.args.input_path, "npy", f"{row.ID}_mask.npy").replace(" ","")
        image = np.load(img_dir) # T, H, W 
        mask = np.load(mask_dir) # T, H, W
        image = np.transpose(image, (1,2,0)) # H, W, T
        mask = np.transpose(mask, (1,2,0)) # H, W, T
        
        if self.args.cut_z < 1:
            num_slices = image.shape[2]
            cut_slices = int(num_slices*self.args.cut_z)
            image=image[:,:,:cut_slices]
            mask=mask[:,:,:cut_slices]            

        data=dict()
        data["image"]=image[np.newaxis,:,:,:] # C, H, W, T
        data["mask"]=mask[np.newaxis,:,:,:] # C, H, W, T
        img_shape = image.shape
        transform = self.get_transform(img_shape, mode=self.mode)
        data = apply_transform(transform, data)
        return data
        
    def get_transform(self, image_size, mode='train'):
        if mode=="train":
            transform = Compose([
                RandSpatialCropd(keys=["image","mask"], roi_size=[int(image_size[0]*self.args.rand_crop),
                                                                   int(image_size[1]*self.args.rand_crop),
                                                                   int(image_size[2]*self.args.rand_crop)], random_size=False),
                Resized(keys=["image","mask"],spatial_size=(self.args.resize_h, self.args.resize_w, self.args.resize_t),mode = "area"),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                ScaleIntensityd(keys=["image"]), # Change pixels range 0 to 1
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),    
                ToTensord(keys=["image", "mask"]),
                ])
        else:
            transform = Compose([
                Resized(keys=["image", "mask"],spatial_size=(self.args.resize_h, self.args.resize_w, self.args.resize_t),mode = "area"),
                ScaleIntensityd(keys=["image"]), # Change pixels range 0 to 1
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),   
                ToTensord(keys=["image", "mask"]),
                ])
        return transform
    
def train_epoch(model, loader, optiminzer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for data in bar:
        images = data["image"].to(device) # B, C, H, W, T 
        masks = data["mask"].to(device) # B, C, H, W, T
        optimizer.zero_grad()
        outputs = model(images)
        dice_loss = criterion(outputs, masks)
        if not args.bce_lambda > 0:
            bce_loss = F.binary_cross_entropy_with_logits(outputs, masks)
            total_loss = dice_loss + bce_loss * args.bce_lambda
        else:
            total_loss = dice_loss
        if torch.cuda.is_available():
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        total_loss = total_loss.detach().cpu().item()
        train_loss.append(total_loss)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (total_loss, smooth_loss))
        del images, masks, outputs, dice_loss
    loss_mean = np.mean(train_loss)
    return loss_mean

    
def get_iou_dice(preds, targets, SMOOTH = 1e-6):
    preds = preds>0.5
    targets = targets.type(torch.bool)
    intersection = (preds & targets).float().sum((1, 2, 3))
    union = (preds | targets).float().sum((1, 2, 3)) 
    mask_sum = (preds.float() + targets.float()).sum((1, 2, 3))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice = (2 * (intersection + SMOOTH)) / (mask_sum + SMOOTH)
    return iou, dice

def val_epoch(model, loader):
    model.eval()
    val_loss={}

    bar = tqdm(loader)
    val_loss = []
    val_dice = []
    with torch.no_grad():
        for data in bar: #B, C, H, W, T
            images = data['image'].to(device) # B, C, H, W, T 
            masks = data["mask"].to(device) # B, C, H, W, T
            logits = model(images)
            dice_loss = criterion(logits, masks)
            if not args.bce_lambda > 0:
                bce_loss = F.binary_cross_entropy_with_logits(outputs, masks)
                total_loss = dice_loss + bce_loss * args.bce_lambda
            else:
                total_loss = dice_loss
            val_loss.append(total_loss.detach().cpu().numpy().item())
            probs = logits.sigmoid().detach().cpu()
            masks = masks.detach().cpu()
            del logits
            probs[probs > 0.5] = 1
            probs[probs <= 0.5] = 0
            probs = probs.type(torch.uint8)
            _, dice = get_iou_dice(probs.squeeze(1), masks.squeeze(1), SMOOTH = 1e-6)
            dice = dice.mean().item()
            val_dice.append(dice)
            del images, masks, dice_loss, probs, dice
        val_loss = np.mean(val_loss)
        val_dice = np.mean(val_dice)
    return val_loss, val_dice

def plot_loss(csv_dir, png_dir):
    plt.rcParams["figure.figsize"]=(5,5)
    fig = plt.figure(figsize=(10,8))
    csv_df = pd.read_csv(csv_dir)
    train_loss = csv_df[f'train_loss'].to_list()
    valid_loss = csv_df[f'val_loss'].to_list()
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Min Loss')

    plt.xlabel('epochs', fontsize=14)
    plt.ylabel(f'loss', fontsize=14)
    plt.xlim(0, len(train_loss)+1) 
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(png_dir, bbox_inches = 'tight', pad_inches = 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default= 410)
    parser.add_argument('--input_path', type=str, default='../data/process/total/')
    parser.add_argument('--output_path', type=str, default='../outputs/')
    parser.add_argument('--suffix', type=str, default='debug')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--DEBUG', type=str, default="T")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_dir", type=str, default="../outputs/test_220117/best_dice01.pth") 
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--cut_z", type=float, default=0.6)
    parser.add_argument("--rand_crop", type=float, default=0.95)
    parser.add_argument("--resize_h", type=int, default=64) # 96, 128 160
    parser.add_argument("--resize_w", type=int, default=64) # 96, 128 160
    parser.add_argument("--resize_t", type=int, default=32) # 96, 64 160
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument('--plt_show', type=str, default="T")
    parser.add_argument('--model', type=str, default="segresnet") # unet, segresnet, unetr
    parser.add_argument("--bce_lambda", type=float, default=0.5)
    args, _ = parser.parse_known_args()
    print(args)
    
set_seed(seed=args.SEED)
output_path = os.path.join(args.output_path, args.suffix)
os.makedirs(output_path, exist_ok=True)

set_dir = os.path.join(output_path, f'train_set{args.fold:02d}.txt')
with open(set_dir, 'w') as appender: 
    appender.write(str(args) + '\n')  
    
csv_dir = os.path.join(output_path, f'train_log{args.fold:02d}.csv')
with open(csv_dir, 'w') as appender:
    appender.write("Epochs,lr,train_loss,val_loss,val_dice" + '\n')  

df = pd.read_csv(os.path.join(args.input_path,"total_data.csv"))
df['fold'] = df.index % 5 + 1 

df_train = df[(df['fold'] != args.fold)].reset_index(drop=True) 
df_val = df[(df['fold'] == args.fold)].reset_index(drop=True)

df_train = df_train.sample(frac = 1).reset_index(drop=True) 
df_val = df_val.sample(frac = 1).reset_index(drop=True) 

if args.DEBUG == "T":
    print('DEBUGING...')
    df_train = df_train[0:5]
    df_val = df_val[0:5]
    args.n_epochs = 3

dataset_train = MandibleData(df_train, args, "train")  
dataset_val = MandibleData(df_val, args, "val")     

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

if args.plt_show=="T":
    rcParams['figure.figsize'] = 15,15
    f, axarr = plt.subplots(2,4)
    for idx in range(4):
        data = loader_train.dataset[idx]
        print(data['image'].shape)
        for j, img in enumerate(['image', 'mask']):
            data[img] = data[img].squeeze().sum(axis=0)
            axarr[j, idx].imshow(data[img], cmap='gray')
            axarr[j, idx].axis("off")
            axarr[j, idx].set_title(f"Data {idx}: {img}")
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available(): scaler = torch.cuda.amp.GradScaler()
if args.model == 'unet':
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
elif args.model == 'segresnet':
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.2,
        ).to(device)
elif args.model == 'unetr':
    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(args.resize_h, args.resize_w, args.resize_t),
        feature_size=16, # 16
        hidden_size=768, # 768
        mlp_dim=3072, # 3072
        num_heads=12, # 12
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        ).to(device)
    
if args.weight_dir is not None: 
    print(f"Load '{args.weight_dir}'...")
    try:  # single GPU model_file
        model.load_state_dict(torch.load(args.weight_dir), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(args.weight_dir)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
contents = f"Model: {args.model} // Total parameter: {format(total_params, ',d')} // Trainable parameter: {format(trainable_params, ',d')}"
print(contents)
with open(set_dir, 'a') as appender: appender.write(contents+"\n")  

criterion = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs)
val_dice_best = -1
val_loss_best = np.inf
early_count = 0
print(len(dataset_train), len(dataset_val))

for epoch in range(args.n_epochs):
    scheduler_cosine.step(epoch)
    train_loss = train_epoch(model, loader_train, optimizer)
    val_loss, val_dice = val_epoch(model, loader_val)
    content = f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.5f} trn loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val dice: {val_dice:.4f}'
    print(content)
    with open(csv_dir, 'a') as appender:
        lr = optimizer.param_groups[0]["lr"]
        appender.write(f"{epoch},{lr:.6f},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")  
    torch.save(model.state_dict(), os.path.join(output_path,f'last_epoch{args.fold:02d}.pth'))
    if val_loss < val_loss_best:
        print('val_loss_best ({:.5f} --> {:.5f})'.format(val_loss_best, val_loss))
        torch.save(model.state_dict(), os.path.join(output_path,f'best_loss{args.fold:02d}.pth'))
        val_loss_best = val_loss
    if val_dice > val_dice_best:
        print('val_dice_best ({:.5f} --> {:.5f})'.format(val_dice_best, val_dice))
        torch.save(model.state_dict(), os.path.join(output_path,f'best_dice{args.fold:02d}.pth'))
        val_dice_best = val_dice
        early_count = 0
    else:
        early_count += 1
        if args.early_stop != 0:
            print(f'EarlyStopping counter: {early_count} out of {args.early_stop}')
        if early_count == args.early_stop:
            print("Early stopping")
            break
            
png_dir = os.path.join(output_path, "epochs_loss.png")
plot_loss(csv_dir, png_dir)
