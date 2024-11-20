from torch import Tensor
import torch.nn.functional as F

import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset

def classwise_dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):

    assert input.size() == target.size()
    num_classes = input.size(1)

    class_dice_scores = []
    for class_idx in range(num_classes):
        # Extract the binary masks for the current class
        input_class = input[:, class_idx, :, :]
        target_class = target[:, class_idx, :, :]

        # Compute Dice for this class
        inter = 2 * (input_class * target_class).sum(dim=(-1, -2))
        sets_sum = input_class.sum(dim=(-1, -2)) + target_class.sum(dim=(-1, -2))
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        class_dice_scores.append(dice.mean().item())

    return class_dice_scores

def validate(net, dataloader, device, num_classes, mask_values, epsilon=1e-6):

    net.eval()
    class_dice_scores = torch.zeros(num_classes, device=device)

    num_val_batches = len(dataloader)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            # Convert predictions to one-hot format
            mask_pred = torch.softmax(mask_pred, dim=1)  # probabilities for each class
            mask_pred_onehot = mask_pred.argmax(dim=1, keepdim=True)
            mask_pred_onehot = torch.nn.functional.one_hot(mask_pred_onehot.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Convert ground truth to one-hot format
            mask_true_onehot = torch.nn.functional.one_hot(mask_true, num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Compute class-wise Dice scores for the batch
            batch_dice_scores = torch.tensor(classwise_dice_coeff(mask_pred_onehot, mask_true_onehot, epsilon=epsilon), device=device)

            class_dice_scores += batch_dice_scores

    class_dice_scores /= num_val_batches

    return class_dice_scores.cpu().tolist()

if  __name__  == '__main__':
    img_scale = 0.3
    batch_size = 4
    
    dir_img = Path('./data/val/imgs/')
    dir_mask = Path('./data/val/masks/')
    
    val_set = BasicDataset(dir_img, dir_mask, img_scale)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    dataloader = DataLoader(val_set, shuffle=False, drop_last=True)
    
    net = UNet(n_channels=3, n_classes=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.to(device=device)
    state_dict = torch.load("./checkpoints/checkpoint_epoch5.pth", map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    dice_scores = validate(
    net=net,
    dataloader=dataloader,
    device=device,
    num_classes=10,
    mask_values=mask_values
    )
    print("Overall Dice scores for each class:", dice_scores)
