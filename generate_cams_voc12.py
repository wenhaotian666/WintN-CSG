# -*- coding:UTF-8 -*-
from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os


from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip_text import class_names, new_class_names, BACKGROUND_CATEGORY#, imagenet_templates
import argparse
from lxml import etree
import time
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, n_splits):    #切片数据集
    if n_splits == 1:
        return [dataset]
    part = len(dataset)//n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])
    return dataset_list

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #列表的每个元素就是'a clean origami {背景类名}.'
            texts = clip.tokenize(texts).to(device) #tokenize将文本转化为token
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)#标准化
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)#标准化之后的文本嵌入向量列表
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)#将所有类别的嵌入向量堆叠成一个张量，就是一个只有一列以及classnames数量行的列表
    return zeroshot_weights.t().half()#返回转置后的权重张量，这样每一列对应一个类别。

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w): #
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

#水平翻转则是一种简单的数据增强技术，可以增加数据的多样性，提升模型的泛化能力。
def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16): #得到原始图像和翻转过后图像的图像集合
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))#ceil是向上取整
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])#水平反转，比如1234变成4321
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


def perform(process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam): #0，datalist=trainlist，
    #准备gpu设备
    n_gpus = torch.cuda.device_count()
    device_id = "cuda:{}".format(process_id % n_gpus)
    model = model.to(device_id)
    databin = dataset_list[process_id] #dataset=trainlist也就是训练图像列表
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)

    #im_idx是索引,im是图像文件，这段代码的作用是为每张图像构建对应的 XML 文件路径，这些 XML 文件可能包含了图像的标注信息，例如对象的位置、类别等。这些标注信息通常用于训练或评估计算机视觉模型。
    for im_idx, im in enumerate(tqdm(databin)):
        img_path = os.path.join(args.img_root, im)#取图像文件路径
        xmlfile = img_path.replace('/JPEGImages', '/Annotations')
        xmlfile = xmlfile.replace('.jpg', '.xml')

        #取xml文件里annotation字典里中的子字典，以及其中字典键对应的值
        with open(xmlfile) as fid:#打开名为 xmlfile 的文件，并将文件对象赋值给 fid
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)  # etree包 读取xml文件 将字符串形式的XML数据解析为一个 etree 元素对象，赋值给 xml
        data = parse_xml_to_dict(xml)["annotation"]
        ori_width = int(data['size']['width']) #找data字典里size键对应的子字典中的width键所对应的值赋值给ori_width,也就是对应图片的初始大小
        ori_height = int(data['size']['height'])

        #这一步就用于更新clip所运用的类文本,这一步就是同义词融合
        label_list = []
        label_id_list = []
        for obj in data["object"]:#找data字典里object键对应的子字典中的name键所对应的值,
            obj["name"] = new_class_names[class_names.index(obj["name"])]#找到name所对应的值在class_names里对应的索引，然后取new_class_names中对应这个索引的值复制给name所对应的值，更新name键所对应的值，索引为14
            if obj["name"] not in label_list:
                label_list.append(obj["name"])#如果这个name对应的值不在里面就把他加进去，索引也加进去
                label_id_list.append(new_class_names.index(obj["name"]))#

        if len(label_list) == 0:
            print("{} not have valid object".format(im))
            return

        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0]) #返回一个多尺度图像列表，有两个尺度，第一个是原始图像列表，第二个是水平翻转列表
        ms_imgs = [ms_imgs[0]]  #这个步骤的目的是只保留一个尺度的图像，而不使用多尺度的图像，就是说只取了第一个原始图像列表
        cam_all_scales = []
        highres_cam_all_scales = []
        refined_cam_all_scales = []


        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)
            image_features, attn_weight_list = model.encode_image(image, h, w)#得到图像特征和注意力列表

            cam_to_save = []
            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = bg_text_features.to(device_id)  # [bg_id_for_each_image[im_idx]].to(device_id)
            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)#把他俩在维度上拼接，第一个维度 行上
            input_tensor = [image_features, text_features_temp.to(device_id), h, w]#vit出来的图像特征，背景和前景在行上拼接的特征，高，宽

            for idx, label in enumerate(label_list):
                keys.append(new_class_names.index(label))
                targets = [ClipOutputTarget(label_list.index(label))] #targets就是ClipOutputTarget的一个实例


                #torch.cuda.empty_cache()
                #grayscale_cam：灰度化的类激活图（Class Activation Map），表示每个像素点对于目标类别的重要性。logits_per_image：每个图像的原始预测 logits（对数几率）。attn_weight_last：最后一层的注意力权重
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)  # (ori_width, ori_height))

                grayscale_cam = grayscale_cam[0, :]


                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))#使用 OpenCV 的 cv2.resize 函数将灰度化类激活图 grayscale_cam 调整为原始输入图像的尺寸 (ori_width, ori_height)
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres)) #这个列表存储了每个输入图像对应的高分辨率灰度化类激活图。

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)，第一个是clstoken所以要去除
                    attn_weight = torch.stack(attn_weight, dim=0)[-6:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()#得到注意力权重

                #CAA
                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                box_aff = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    box_aff[y0_:y1_, x0_:x1_] = 1

                box_aff = box_aff.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight


                #对注意力权重交替使用行列归一化
                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True) #D

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2 #affinity matrix A

                for _ in range(1):
                     trans_mat = torch.matmul(trans_mat, trans_mat) #A方 t=2 细化两次

                trans_mat = trans_mat * box_aff #At*Bc (576,576)

                cam_to_refine = torch.FloatTensor(grayscale_cam)#Mc
                cam_to_refine = cam_to_refine.view(-1,1)


                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16) #At*Bc*Mc
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))



            keys = torch.tensor(keys) #保存新类名的索引
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))  #高分辨率灰度图
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))  #细化后的掩码cam


        #cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        np.save(os.path.join(args.cam_out_dir, im.replace('jpg', 'npy')),
                {"keys": keys.numpy(),
                # "strided_cam": cam_per_scales.cpu().numpy(),
                #"highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
                "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),#细化后的掩码
                })
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_root', type=str, default='F:/Studyfiles/CLIP-ES-main/datasets/VOC2012/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./voc12/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./final/ablation/baseline')
    parser.add_argument('--model', type=str, default='./pretrained_models/clip/ViT-B-16.pt')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + '.jpg' for x in train_list] #图像文件列表

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, _ = clip.load(args.model, device=device)
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], model)#['a rendering of a weird {}.'], model)
    fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], model)#['a rendering of a weird {}.'], model)

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)


    #datalist就是trainlist，也就是训练图像列表
    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_list, args, model, bg_text_features, fg_text_features, cam))

