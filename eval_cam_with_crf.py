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

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

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


def refine_label(label):
    # 获取标签矩阵的高度和宽度
    h, w = label.shape

    # 创建一个新的标签矩阵来存储结果
    refined_label = label.copy()

    # 中值滤波+形态学去噪
    refined_label = refined_label.astype(np.uint8)

    # 定义核
    kernel = np.ones((3, 3), np.uint8)

    # 应用开运算
    refined_label = cv2.morphologyEx(refined_label, cv2.MORPH_OPEN, kernel)
    refined_label = cv2.medianBlur(refined_label, 5)

    # 遍历每个像素（不包括边界像素
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 当前像素值
            current_pixel = refined_label[i, j]

            # 上下左右四个像素值
            top_pixel = refined_label[i - 1, j]
            bottom_pixel = refined_label[i + 1, j]
            left_pixel = refined_label[i, j - 1]
            right_pixel = refined_label[i, j + 1]

            # 如果当前像素不为0且上下左右四个像素中至少有三个为0
            if current_pixel != 0 and top_pixel== 0 and bottom_pixel== 0 and left_pixel== 0 and right_pixel== 0 :
                # 将当前像素值设置为0
                refined_label[i, j] = 0

    return refined_label

def crf(n_jobs, is_coco=False):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Process per sample
    def process(i):
        image_id = eval_list[i]
        image_path = os.path.join(args.image_root, image_id + '.jpg')###
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label_path = os.path.join(args.gt_root, image_id + '.png')###[15:]
        gt_label = np.asarray(Image.open(label_path), dtype=np.int32)

        # Mean subtraction
        sum_b = 0
        sum_g = 0
        sum_r = 0
        count = 0

        if image is not None:
            # 累加每个通道的值
            sum_b += np.sum(image[:, :, 0])
            sum_g += np.sum(image[:, :, 1])
            sum_r += np.sum(image[:, :, 2])
            count += image.shape[0] * image.shape[1]

        # 计算每个通道的均值
        mean_b = sum_b / count
        mean_g = sum_g / count
        mean_r = sum_r / count

        mean_bgr = (mean_b, mean_g, mean_r)
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        filename = os.path.join(args.cam_out_dir, image_id + ".npy")###
        cam_dict = np.load(filename, allow_pickle=True).item()
        cams = cam_dict['attn_highres']
        class_idx = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        #CAL
        background_prob = 1 - np.max(cams, axis=0, keepdims=True)
        total_probmap = np.concatenate((background_prob, cams), axis=0)
        image = image.astype(np.uint8).transpose(1, 2, 0)
        probmap = postprocessor(image, total_probmap)

        idx_perpix = np.argmax(probmap, axis=0)
        label = class_idx[idx_perpix]
        label = refine_label(label)

        if not args.eval_only:
            activation = np.max(probmap, axis=0)
            label[activation < 0.95] = 255
            cv2.imwrite(os.path.join(args.pseudo_mask_save_path, image_id + '.png'), label.astype(np.uint8))  ###

        return label.astype(np.uint8), gt_label.astype(np.uint8)

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(eval_list))]
    )
    if args.eval_only:###
        preds, gts = zip(*results)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=21 if not is_coco else 81)
        print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./output/voc12/cams", type=str)
    parser.add_argument("--pseudo_mask_save_path", default="./output/voc12/pseudo_masks_without_crf", type=str)
    parser.add_argument("--split_file", default="./voc12/try.txt",
                        type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/SegmentationClassAug", type=str)
    parser.add_argument("--image_root", default="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/JPEGImages", type=str)
    parser.add_argument("--eval_only", default=False,action="store_true")
    args = parser.parse_args()

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    is_coco = 'coco' in args.cam_out_dir
    if 'voc' in args.cam_out_dir:
        eval_list = list(np.loadtxt(args.split_file, dtype=str))
    elif 'coco' in args.cam_out_dir:
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        eval_list = [x[0] for x in file_list]#[:2000]
    print('{} images to eval'.format(len(eval_list)))

    if not args.eval_only and not os.path.exists(args.pseudo_mask_save_path):
        os.makedirs(args.pseudo_mask_save_path)

    n_jobs =multiprocessing.cpu_count()
    crf(n_jobs, is_coco)

# split_file= "./voc12/train_aug.txt"
# eval_list = list(np.loadtxt(split_file, dtype=str))
# mean_bgr = (104.008, 116.669, 122.675)
# n_jobs =multiprocessing.cpu_count()
# is_coco=False
# cam_out_dir="./output/voc12/cams"
# gt_root ="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/SegmentationClassAug"
# image_root ="F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/JPEGImages"
# pseudo_mask_save_path="./output/voc12/pseudo_masks_without_crf"
# eval_only = False
# crf(n_jobs, is_coco)