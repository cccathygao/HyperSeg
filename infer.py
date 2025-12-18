import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
import cv2
import re
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import SiglipImageProcessor

from hyperseg.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX
from hyperseg.utils.builder import load_pretrained_model
from hyperseg.utils import conversation as conversation_lib
from hyperseg.eval.eval_dataset.eval_datasets import DataCollatorForCOCODatasetV2, RefCOCO_dataset_test

from pycocotools import mask


@dataclass
class DataArguments:
    local_rank: int = 0
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    
    vision_tower: str = "./pretrained_model/siglip-so400m-patch14-384"
    vision_tower_mask: str = "./pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default='../referring-segmentation') # './dataset/coco/train2014'
    model_path: Optional[str] = field(default="./model/HyperSeg-3B")
    mask_config: Optional[str] = field(default="./hyperseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    json_path: str = './dataset/refcoco_evaluation.json'
    model_map_name: str = 'HyperSeg'
    version: str = 'llava_phi'
    output_dir: str = './output_hyperseg'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 0
    seg_task: Optional[str] = field(default="referring")
    sum_ref_pred_answer: bool = False
    enable_mgvp_seg_query: bool = field(default=True)
    visualize: bool = True
    
    # New arguments for batch processing
    image_list: Optional[str] = field(default=None)
    image_id: Optional[str] = field(default=None)
    dataset_json: str = '../referring-segmentation/grefcoco_dataset/dataset.json'


def parse_segmentation_string(seg_string):
    """Parse the segmentation string format to extract polygon coordinates."""
    pattern = r'<seg>(.*?)</seg>'
    match = re.search(pattern, seg_string)
    if not match:
        return None
    
    coords_string = match.group(1)
    coord_pattern = r'\(([^)]+)\)'
    coords = re.findall(coord_pattern, coords_string)
    
    polygon = []
    for coord in coords:
        x, y = coord.split(',')
        polygon.append([float(x.strip()), float(y.strip())])
    
    return np.array(polygon)


def polygon_to_mask(polygon, height, width):
    """Convert polygon coordinates to binary mask."""
    mask_np = np.zeros((height, width), dtype=np.uint8)
    if polygon is not None and len(polygon) > 0:
        polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask_np, [polygon], 1)
    return mask_np.astype(bool)


def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    polygon = []
    for point in largest_contour:
        x, y = point[0]
        polygon.append([float(x), float(y)])
    
    return polygon


def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def process_single_image(model, tokenizer, clip_image_processor, data_collator, dataset, image_id, data_args, device, mask_output_dir):
    """Process a single image and return IoU."""
    
    if image_id not in dataset:
        print(f"Error: Image ID '{image_id}' not found in dataset.")
        return None
    
    data_entry = dataset[image_id]
    prompt = data_entry["problem"]
    image_path = data_entry["images"][0]
    gt_answer = data_entry["answer"]
    img_height = data_entry["img_height"]
    img_width = data_entry["img_width"]
    
    print(f"\n{'='*60}")
    print(f"Processing image ID: {image_id}")
    print(f"Prompt: {prompt}")
    print(f"Image path: {image_path}")

    relative_image_path = '../referring-segmentation/' + image_path
    
    if not os.path.exists(relative_image_path):
        print(f"Error: File not found at {relative_image_path}")
        return None
    
    # Parse ground truth mask
    gt_polygon = parse_segmentation_string(gt_answer)
    gt_mask = polygon_to_mask(gt_polygon, img_height, img_width)
    gt_mask_tensor = torch.tensor(gt_mask.astype(np.uint8), device=device)
    
    # Create evaluation data entry in the format expected by RefCOCO_dataset_test
    eval_entry = {
        'new_img_id': image_id,
        'image_info': {
            'file_name': image_path,
            'height': img_height,
            'width': img_width,
            'id': image_id
        },
        'instruction': [{'raw': prompt}],
        'anns': [{
            'segmentation': [gt_polygon.flatten().tolist()],
            'area': int(gt_mask.sum()),
            'bbox': [0, 0, img_width, img_height]
        }]
    }
    
    # Create a temporary single-item dataset
    temp_json_path = os.path.join(mask_output_dir, f'temp_{image_id}.json')
    with open(temp_json_path, 'w') as f:
        json.dump([eval_entry], f)
    
    try:
        # Create dataset and dataloader for this single image
        eval_dataset = RefCOCO_dataset_test(json_path=temp_json_path, tokenizer=tokenizer, data_args=data_args)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=data_collator
        )
        
        # Run inference
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                inputs['token_refer_id'] = [ids.to(device) for ids in inputs['token_refer_id']]
                
                outputs = model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    images_clip=inputs['images_clip'].float(),
                    seg_info=inputs['seg_info'],
                    token_refer_id=inputs['token_refer_id'],
                    refer_embedding_indices=inputs['refer_embedding_indices'],
                    labels=inputs['labels']
                )
                
                # Process outputs
                best_iou = 0.0
                best_pred_mask = None
                
                for output in outputs:
                    pred_masks = output['instances'].pred_masks
                    scores = output['instances'].scores
                    
                    # Pick mask with maximum score
                    topk_scores, idx = torch.topk(scores, 1)
                    topk_pred = pred_masks[idx[0]]
                    
                    # Calculate IoU
                    intersection, union, _ = intersectionAndUnionGPU(
                        topk_pred.int().contiguous().clone(), 
                        gt_mask_tensor.int().contiguous(), 
                        2, 
                        ignore_index=255
                    )
                    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                    acc_iou = intersection / (union + 1e-5)
                    acc_iou[union == 0] = 1.0
                    fore_acc_iou = acc_iou[1]
                    
                    if fore_acc_iou > best_iou:
                        best_iou = fore_acc_iou
                        best_pred_mask = topk_pred.detach().cpu().numpy()
                
                print(f"Best IoU: {best_iou:.4f}")
                
                # Save visualizations and JSON
                if best_pred_mask is not None:
                    pred_mask_binary = best_pred_mask > 0
                    
                    # Resize if needed
                    if pred_mask_binary.shape != (img_height, img_width):
                        pred_mask_binary = cv2.resize(
                            pred_mask_binary.astype(np.uint8),
                            (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # Save masked image
                    image_np = cv2.imread(relative_image_path)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    
                    save_img = image_np.copy()
                    save_img[pred_mask_binary] = (
                        image_np * 0.5
                        + pred_mask_binary[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask_binary]
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    
                    vis_path = os.path.join(data_args.output_dir, f'{image_id}_masked.jpg')
                    cv2.imwrite(vis_path, save_img)
                    print(f"Saved visualization to: {vis_path}")
                    
                    # Save mask image
                    mask_img_path = os.path.join(data_args.output_dir, f'{image_id}_mask.png')
                    cv2.imwrite(mask_img_path, pred_mask_binary.astype(np.uint8) * 255)
                    
                    # Convert mask to polygon and save JSON
                    pred_polygon = mask_to_polygon(pred_mask_binary)
                    
                    output_json = {
                        "images": [{
                            "id": image_id,
                            "file_path": image_path,
                            "data_source": "",
                            "height": img_height,
                            "width": img_width,
                            "scene": "",
                            "is_longtail": False,
                            "task": "referring_segmentation",
                            "problem": prompt,
                            "problem_type": {
                                "num_class": "",
                                "num_instance": ""
                            }
                        }],
                        "annotations": [{
                            "id": 0,
                            "image_id": image_id,
                            "category_id": None,
                            "bbox": None,
                            "area": None,
                            "shape_type": "polygon",
                            "error_type": None,
                            "iou": float(best_iou),
                            "segmentation": [pred_polygon]
                        }]
                    }
                    
                    json_output_path = os.path.join(mask_output_dir, f"{image_id}_mask.json")
                    with open(json_output_path, 'w') as f:
                        json.dump(output_json, f, indent=4)
                    print(f"Saved mask JSON to: {json_output_path}")
                    
                    return best_iou
                else:
                    print("No valid predicted mask found.")
                    return None
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)


def main():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    
    # Create output directories
    os.makedirs(data_args.output_dir, exist_ok=True)
    mask_output_dir = "./mask_output"
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # Load dataset JSON
    with open(data_args.dataset_json, 'r') as f:
        dataset = json.load(f)
    
    # Get list of image IDs to process
    image_ids = []
    if data_args.image_list:
        with open(data_args.image_list, 'r') as f:
            content = f.read()
            image_ids = [id.strip().strip("'\"") for id in content.split(',') if id.strip()]
        print(f"Loaded {len(image_ids)} image IDs from {data_args.image_list}")
    elif data_args.image_id:
        image_ids = [data_args.image_id]
    else:
        print("Error: Must provide either --image_id or --image_list")
        return
    
    # Load model
    print("Loading model...")
    model_path = os.path.expanduser(data_args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_args=data_args, 
        mask_config=data_args.mask_config, 
        device='cuda'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dtype=torch.float32, device=device)
    model.eval()
    
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    data_args.refcoco_image_folder = data_args.image_folder
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    
    print("Model loaded successfully!")
    
    # Process all images
    results = []
    successful = 0
    failed = 0
    
    for idx, image_id in enumerate(image_ids):
        print(f"\n{'#'*60}")
        print(f"Progress: {idx + 1}/{len(image_ids)}")
        print(f"{'#'*60}")
        
        try:
            iou = process_single_image(
                model, tokenizer, clip_image_processor, data_collator,
                dataset, image_id, data_args, device, mask_output_dir
            )
            if iou is not None:
                results.append((image_id, iou))
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(image_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        avg_iou = sum(iou for _, iou in results) / len(results)
        print(f"\nAverage IoU: {avg_iou:.4f}")
        print(f"Best IoU: {max(iou for _, iou in results):.4f}")
        print(f"Worst IoU: {min(iou for _, iou in results):.4f}")
        
        # Save summary
        summary_path = os.path.join(mask_output_dir, "summary.json")
        summary = {
            "total_images": len(image_ids),
            "successful": successful,
            "failed": failed,
            "average_iou": float(avg_iou),
            "results": [{"image_id": img_id, "iou": float(iou)} for img_id, iou in results]
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()