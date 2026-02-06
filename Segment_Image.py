import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import re

import numpy as np
import json
import torch
from PIL import Image
from llm_agent import chat_with_llm

sys.path.append(os.path.join(os.getcwd(), "GroundSAM/GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "GroundSAM/segment_anything"))


# Grounding DINO
import GroundSAM.GroundingDINO.groundingdino.datasets.transforms as T
from GroundSAM.GroundingDINO.groundingdino.models import build_model
from GroundSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def get_relevance_scores(objects, target_text, target_image_path):

    objects_list = [item.strip() for item in objects.split('.') if item.strip()]
    prompt = (f"You are looking for the ['{target_text}'] in the image. "
            f"Please analyze the relevance of the following {len(objects_list)} objects('{objects}') to the search target and give a score between 0 and 1 (rounded to two decimal places)."
            "When analyzing the relevance, consider in sequence whether the search target is likely to exist in the object/scene to be scored."
            "0 indicates completely impossible, and 1 indicates highly likely."
            "Scores are required to be evenly distributed between 0 and 1."
            "Only return the score numbers, separated by commas, without any other words")


    # 调用LLM获取响应
    response = chat_with_llm(prompt, ("./" + target_image_path))


    # 处理响应文本，转换为浮点数列表
    try:
        scores = [float(score.strip()) for score in response.split(',')]

        scores = [max(0.0, min(1.0, score)) for score in scores]


        if len(scores) != len(objects_list):
            raise ValueError("返回的分数数量与类别数量不匹配")


        objects_scores = dict(zip(objects_list, scores))

        return objects_scores

    except Exception as e:
        print(f"处理LLM响应时出错: {e}")
        return None


def overlay_masks_on_depth(depth_image, masks, class_ids, class_scores):
    """
    将masks叠加到深度图上，每个mask乘以其索引值

    Args:
        depth_image (np.ndarray): 原始深度图
        masks (Tensor or np.ndarray): 检测到的masks数组 (可能是 GPU 上的 Tensor)
        class_ids (list/array): 类别ID
        class_scores (list/array): 类别置信度

    Returns:
        np.ndarray: 叠加masks后的深度图
    """
    # 初始化全0矩阵，形状与 depth_image 相同 (480, 640)
    # 建议明确指定类型，防止类型冲突，这里假设用 float32 方便后续计算
    masks_enhanced_id = np.zeros((480, 640), dtype=np.float32)
    masks_enhanced_score = np.zeros((480, 640), dtype=np.float32)

    # 遍历所有masks
    for idx, mask in enumerate(masks):
        # --- 修改 1: 将 Tensor 转换为 Numpy ---
        if isinstance(mask, torch.Tensor):
            # .cpu() 移到内存, .numpy() 转格式, .squeeze() 去掉多余的维度 (1, 480, 640) -> (480, 640)
            mask = mask.cpu().numpy().squeeze()

        # 确保 mask 是布尔型或 0/1 整数，然后转为数值型以便乘法
        mask_val = mask.astype(np.float32)

        # --- 修改 2: 计算逻辑 ---
        # 这里的逻辑是：保留当前像素位置上 ID/Score 最大的那个值

        # 计算当前 mask 对应的 ID 值 (加1是为了避免0值，或者根据你的逻辑调整)
        current_id_map = mask_val * (class_ids[idx] + 1)
        masks_enhanced_id = np.maximum(masks_enhanced_id, current_id_map)

        # 计算当前 mask 对应的 Score 值
        current_score_map = mask_val * (class_scores[idx] + 1)
        # 注意：你原代码这里写成了 masks_enhanced_id，应该是 masks_enhanced_score
        masks_enhanced_score = np.maximum(masks_enhanced_score, current_score_map)

    return masks_enhanced_id, masks_enhanced_score


def get_scores_for_class_names(class_names, scores):
    """
    Args:
        class_name_to_id: (未使用)
        class_names: 模型检测到的类别名列表 (例如 ['central all - star sign', ...])
        scores: 你定义的原始分数字典 (例如 {'Central All-Star sign': 1.0, ...})
    """

    # 1. 定义一个辅助函数：标准化字符串
    # 作用：转小写，去掉所有空格，去掉所有连字符
    def normalize_str(s):
        # 移除空格、连字符、下划线，并转小写
        return re.sub(r'[\s\-_]', '', str(s)).lower()

    # 2. 创建一个新的查找表：{标准化后的名字: 分数}
    # 例如将 'Central All-Star sign' 变成 'centralallstarsign' 作为键
    normalized_scores_map = {}
    for key, value in scores.items():
        clean_key = normalize_str(key)
        normalized_scores_map[clean_key] = value

    scores_names = []

    # 3. 遍历检测到的名字，进行匹配
    for name in class_names:
        # 同样标准化检测到的名字
        # 'central all - star sign' -> 'centralallstarsign'
        clean_name = normalize_str(name)

        if clean_name in normalized_scores_map:
            scores_names.append(normalized_scores_map[clean_name])
        else:
            # 如果实在找不到，给一个默认分 (比如 0.5 或 1.0)，防止程序崩溃
            print(f"Warning: Key '{name}' (normalized: '{clean_name}') not found in scores dict. Using default 0.5.")
            scores_names.append(0.5)

    return scores_names


def segment_observation(image_path, text_prompt, dino_model, sam_predictor, device="cuda"):
    """
    Args:
        image_path: 图片路径
        text_prompt: 文本提示
        dino_model: 预加载的 GroundingDINO 模型
        sam_predictor: 预加载的 SAM Predictor
        device: 设备
    """
    box_threshold = 0.4
    text_threshold = 0.25

    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    with torch.no_grad():
        outputs = dino_model(image_tensor[None].to(device), captions=[text_prompt])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter output
    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]
    pred_phrases = []
    if boxes_filt.shape[0] > 0:
        logits_filt = logits[filt_mask]
        tokenizer = dino_model.tokenizer
        tokenized = tokenizer(text_prompt)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits_filt
        ]
    else:
        print(f"Warning: No objects found for prompt: {text_prompt}")
        return np.array([]), np.array([]), {}, torch.empty((0, 0, 0, 0))

    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    sam_predictor.set_image(image_cv)

    size = image_pil.size
    H, W = size[1], size[0]

    boxes_filt = boxes_filt * torch.tensor([W, H, W, H])  # Scale
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2  # xy = center - wh/2
    boxes_filt[:, 2:] += boxes_filt[:, :2]  # x2y2 = xy + wh

    boxes_filt = boxes_filt.to(device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2])

    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    class_ids_list = []
    class_names_list = []
    unique_name_to_id = {}
    next_id = 0

    for phrase in pred_phrases:
        pred_name = phrase.split('(')[0].strip().lower()
        if pred_name not in unique_name_to_id:
            unique_name_to_id[pred_name] = next_id
            next_id += 1
        class_ids_list.append(unique_name_to_id[pred_name])
        class_names_list.append(pred_name)

    masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
    class_ids_np = np.array(class_ids_list, dtype=int)
    class_names_np = np.array(class_names_list)
    class_name_to_id = {str(name): int(id) for name, id in zip(class_names_np, class_ids_np)}

    torch.cuda.empty_cache()

    return class_ids_np, class_names_np, class_name_to_id, masks


if __name__ == "__main__":

    image_path = "./GroundSAM/assets/demo7.jpg"
    text_prompt = "Horse. Sky"
    segment_observation(image_path, text_prompt)