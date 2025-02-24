import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
from lxml import etree
from utils import parse_xml_to_dict
import argparse
from clip_text import class_names, new_class_names, BACKGROUND_CATEGORY


def split_dataset(dataset, n_splits):    #切片数据集
    if n_splits == 1:
        return [dataset]
    part = len(dataset)//n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])
    return dataset_list


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

#例如，hist[0, 0] 表示类别 0 被正确预测的次数，hist[1, 2] 表示实际为类别 1 被错误预测为类别 2 的次数，明天再看
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def crf(n_jobs, is_coco=False):


    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)


    def process(i):
        # #取gt
        image_id = eval_list[i]
        label_path = os.path.join(args.gt_root, image_id + '.png') #[15:]
        gt_label = np.asarray(Image.open(label_path), dtype=np.int32)

        #SEPL
        sepl_root = "F:\Studyfiles\CLIP-ES-main\output/voc12\pseudo_masks_gd\pseudo_masks_gd_output_voc12_max_iou_imp2"
        pesudo_path = os.path.join(sepl_root, image_id + '.png')
        pesudo_masks = cv2.imread(pesudo_path, cv2.IMREAD_COLOR).astype(np.float16)  # 读取图像并转换为浮点类型
        pesudo_masks = pesudo_masks.transpose(2, 0, 1)
        pesudo_masks = pesudo_masks[0, :, :]
        label = pesudo_masks

        if not args.eval_only:

            cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))

        return label.astype(np.uint8), gt_label.astype(np.uint8)  # label就是掩码


    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(eval_list))]
    )
    #preds就是最后的掩码
    if args.eval_only:
        preds, gts = zip(*results)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=21 if not is_coco else 81)
        print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./output/voc12/cams", type=str)
    parser.add_argument("--pseudo_mask_save_path", default="/home/xxx/code/code48/ablation/usss/voc/val_attn07_crf", type=str)
    parser.add_argument("--split_file", default="./voc12/train.txt",
                        type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/SegmentationClassAug", type=str)
    parser.add_argument("--image_root", default="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/JPEGImages", type=str)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    is_coco = 'coco' in args.cam_out_dir
    if 'voc' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
        # split_file = "./voc12/train.txt"
        # eval_list = list(np.loadtxt(split_file, dtype=str))
    elif 'coco' in args.cam_out_dir:
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        eval_list = [x[0] for x in file_list]#[:2000]
    print('{} images to eval'.format(len(eval_list)))

    if not args.eval_only and not os.path.exists(args.pseudo_mask_save_path):
        os.makedirs(args.pseudo_mask_save_path)
# split_file = "./voc12/train_aug.txt"
# eval_list = list(np.loadtxt(split_file, dtype=str))
# gt_root ="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/SegmentationClassAug"
# mean_bgr = (104.008, 116.669, 122.675)
# n_jobs =multiprocessing.cpu_count()
# is_coco=False
# eval_only = True

    n_jobs = multiprocessing.cpu_count()
    crf(n_jobs, is_coco)