from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from lut_read import *
from networks import SphFuse512
import datasets
from metrics import Evaluator
from saver import Saver

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="G:/Standford2D3D/", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="stanford2d3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to evaluate on.")

parser.add_argument("--load_weights_dir",default='\\weights\\', type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", default=True, help="if set, save the depth maps and point clouds")

settings = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model.pth")
    model_dict = torch.load(model_path)

    # data
    datasets_dict = {"3d60": datasets.ThreeD60,
                     "stanford2d3d": datasets.Stanford2D3D,
                     "matterport3d": datasets.Matterport3D}
    dataset = datasets_dict[settings.dataset]

    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

    test_file_list = fpath.format(settings.dataset, "test")

    test_dataset = dataset(settings.data_path, test_file_list,
                           model_dict['height'], model_dict['width'], is_training=False)
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {
                "sphfuse512": SphFuse512}
    Net = Net_dict[model_dict['net']]

    model = Net(model_dict['layers'], model_dict['height'], model_dict['width'],
                max_depth=test_dataset.max_depth_meters, fusion_type=model_dict['fusion'],
                se_in_fusion=model_dict['se_in_fusion'])

    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    evaluator = Evaluator(settings.median_align)
    evaluator.reset_eval_metrics()
    saver = Saver(load_weights_folder)
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")
    lut512 = read_lut(512)
    lut128 = read_lut(128)
    lut64 = read_lut(64)
    lut32 = read_lut(32)
    # lut16 = read_lut(16)
    # lut8 = read_lut(8)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["normalized_rgb"].to(device)
            outputs = model(equi_inputs,lut512,lut128,lut64,lut32)

            pred_depth = outputs["pred_depth"].detach().cpu()

            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])
            if settings.save_samples:
                saver.save_samples(inputs["rgb"], gt_depth, pred_depth, mask)

    evaluator.print(1000,load_weights_folder)


if __name__ == "__main__":
    main()
