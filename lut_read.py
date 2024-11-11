from scipy import io
import numpy as np
import scipy
import torch

def read_lut(size):
    H=int(size)
    W=2*int(size)
    LUT_mid = np.zeros((H * W, 2), dtype=int)
    LUT_mid_name = r'G:\jingguo\LUT\LUT'+str(size)+'\LUT_mid.mat'
    LUT_mid = torch.from_numpy(scipy.io.loadmat(LUT_mid_name)['LUT']).reshape((H, W, 2))
    lut_mid = torch.zeros((H, W, 2))
    lut_mid[..., 1] = LUT_mid[..., 1]
    lut_mid[..., 0] = LUT_mid[..., 0]
    lut_mid = torch.unsqueeze(lut_mid, dim=0).float()
    lut_mid = lut_mid.cuda()
    LUT_left = np.zeros((H * W, 2), dtype=int)
    LUT_left_name = r'G:\jingguo\LUT\LUT'+str(size)+'\LUT_left.mat'
    LUT_left = torch.from_numpy(scipy.io.loadmat(LUT_left_name)['LUT']).reshape((H,W, 2))
    lut_left = torch.zeros((H, W, 2))
    lut_left[..., 1] = LUT_left[..., 1]
    lut_left[..., 0] = LUT_left[..., 0]
    lut_left = torch.unsqueeze(lut_left, dim=0).float()
    lut_left = lut_left.cuda()
    LUT_right = np.zeros((H * W, 2), dtype=int)
    LUT_right_name = r'G:\jingguo\LUT\LUT'+str(size)+'\LUT_right.mat'
    LUT_right = torch.from_numpy(scipy.io.loadmat(LUT_right_name)['LUT']).reshape((H, W, 2))
    lut_right = torch.zeros((H, W, 2))
    lut_right[..., 1] = LUT_right[..., 1]
    lut_right[..., 0] = LUT_right[..., 0]
    lut_right = torch.unsqueeze(lut_right, dim=0).float()
    lut_right = lut_right.cuda()
    LUT_up = np.zeros((H * W, 2), dtype=int)
    LUT_up_name = r'G:\jingguo\LUT\LUT' + str(size) + '\LUT_up.mat'
    LUT_up = torch.from_numpy(scipy.io.loadmat(LUT_up_name)['LUT']).reshape((H, W, 2))
    lut_up = torch.zeros((H, W, 2))
    lut_up[..., 1] = LUT_up[..., 1]
    lut_up[..., 0] = LUT_up[..., 0]
    lut_up = torch.unsqueeze(lut_up, dim=0).float()
    lut_up = lut_up.cuda()
    LUT_down = np.zeros((H * W, 2), dtype=int)
    LUT_down_name = r'G:\liujingguo\LUT\LUT' + str(size) + '\LUT_down.mat'
    LUT_down = torch.from_numpy(scipy.io.loadmat(LUT_down_name)['LUT']).reshape((H, W, 2))
    lut_down = torch.zeros((H, W, 2))
    lut_down[..., 1] = LUT_down[..., 1]
    lut_down[..., 0] = LUT_down[..., 0]
    lut_down = torch.unsqueeze(lut_down, dim=0).float()
    lut_down = lut_down.cuda()
    LUT_left_up = np.zeros((H * W, 2), dtype=int)
    LUT_left_up_name = r'G:\jingguo\LUT\LUT' + str(size) + '\LUT_left_up.mat'
    LUT_left_up = torch.from_numpy(scipy.io.loadmat(LUT_left_up_name)['LUT']).reshape((H, W, 2))
    lut_left_up = torch.zeros((H, W, 2))
    lut_left_up[..., 1] = LUT_left_up[..., 1]
    lut_left_up[..., 0] = LUT_left_up[..., 0]
    lut_left_up = torch.unsqueeze(lut_left_up, dim=0).float()
    lut_left_up = lut_left_up.cuda()
    LUT_left_down = np.zeros((H * W, 2), dtype=int)
    LUT_left_down_name = r'G:\jingguo\LUT\LUT' + str(size) + '\LUT_left_down.mat'
    LUT_left_down = torch.from_numpy(scipy.io.loadmat(LUT_left_down_name)['LUT']).reshape((H, W, 2))
    lut_left_down = torch.zeros((H, W, 2))
    lut_left_down[..., 1] = LUT_left_down[..., 1]
    lut_left_down[..., 0] = LUT_left_down[..., 0]
    lut_left_down = torch.unsqueeze(lut_left_down, dim=0).float()
    lut_left_down = lut_left_down.cuda()
    LUT_right_up = np.zeros((H * W, 2), dtype=int)
    LUT_right_up_name = r'G:\jingguo\LUT\LUT' + str(size) + '\LUT_right_up.mat'
    LUT_right_up = torch.from_numpy(scipy.io.loadmat(LUT_right_up_name)['LUT']).reshape((H, W, 2))
    lut_right_up = torch.zeros((H, W, 2))
    lut_right_up[..., 1] = LUT_right_up[..., 1]
    lut_right_up[..., 0] = LUT_right_up[..., 0]
    lut_right_up = torch.unsqueeze(lut_right_up, dim=0).float()
    lut_right_up = lut_right_up.cuda()
    LUT_right_down = np.zeros((H * W, 2), dtype=int)
    LUT_right_down_name = r'G:\jingguo\LUT\LUT' + str(size) + '\LUT_right_down.mat'
    LUT_right_down = torch.from_numpy(scipy.io.loadmat(LUT_right_down_name)['LUT']).reshape((H, W, 2))
    lut_right_down = torch.zeros((H, W, 2))
    lut_right_down[..., 1] = LUT_right_down[..., 1]
    lut_right_down[..., 0] = LUT_right_down[..., 0]
    lut_right_down = torch.unsqueeze(lut_right_down, dim=0).float()
    lut_right_down = lut_right_down.cuda()
    lut = [lut_mid, lut_left,lut_left_up, lut_up, lut_right_up,lut_right, lut_right_down, lut_down,   lut_left_down]
    return lut
def lut_batch(batch,lut):
    lut_mid = lut[0].repeat(batch, 1, 1, 1)
    lut_left = lut[1].repeat(batch, 1, 1, 1)
    lut_left_up = lut[2].repeat(batch, 1, 1, 1)
    lut_up = lut[3].repeat(batch, 1, 1, 1)
    lut_right_up = lut[4].repeat(batch, 1, 1, 1)
    lut_right = lut[5].repeat(batch, 1, 1, 1)
    lut_right_down = lut[6].repeat(batch, 1, 1, 1)
    lut_down = lut[7].repeat(batch, 1, 1, 1)
    lut_left_down = lut[8].repeat(batch, 1, 1, 1)

    new_lut = [lut_mid, lut_left,lut_left_up, lut_up, lut_right_up,lut_right, lut_right_down, lut_down, lut_left_down]
    return new_lut
