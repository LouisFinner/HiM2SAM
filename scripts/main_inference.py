import cv2
import gc
import numpy as np
import os
import os.path as osp
import torch
from sam2.build_sam import build_sam2_video_predictor
import tqdm 
import os
import time
import shutil
from PIL import Image


def load_mask(mask_path):
    mask = np.asarray(Image.open(mask_path)).astype(np.float32)
    mask = (mask > 0).astype(np.uint8)
    return mask
    
def load_lasot_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    
    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        x, y, w, h = map(int, line.split(','))
        prompts[fid] = ((x, y, x+w, y+h), 0)
        fid += 1

    return prompts

color = [
    (255, 0, 0),
]
cuda_device = "cuda:0"

current_dir = os.path.dirname(os.path.abspath(__file__))
testing_set = os.path.join(current_dir, "../data/LaSOT/testing_set.txt")
with open(testing_set, 'r') as f:
    test_videos = f.readlines()

exp_name = "him2sam"
model_name = "large"

description='exp_him2sam_lasot'
checkpoint = os.path.join(current_dir, f"../sam2/checkpoints/sam2.1_hiera_{model_name}.pt")

init_mode = 'box'
assert init_mode in ['mask', 'box']


model_cfg = "configs/him2sam/lasot/sam2.1_hiera_l.yaml"
video_folder= os.path.join(current_dir, "../data/LaSOT")
pred_folder = os.path.join(current_dir, f"../results/{exp_name}/{description}_{exp_name}_{model_name}")
os.makedirs(pred_folder, exist_ok=True)
shutil.copy(os.path.join(current_dir,"../sam2/sam2",model_cfg), 
                pred_folder)
print(model_cfg)
save_to_video = True
if save_to_video:
    vis_folder = os.path.join(current_dir, f"../visualization/{exp_name}/{model_name}_{description}")
    os.makedirs(vis_folder, exist_ok=True)
    vis_mask = {}
    vis_bbox = {}


print("running on cuda:{}, ".format(cuda_device,))
with open(testing_set, 'r') as f:
    test_videos = f.readlines()

for vid, video in enumerate(test_videos):

    cat_name = video.split('-')[0]
    cid_name = video.split('-')[1]
    video_basename = video.strip()
    frame_folder = osp.join(video_folder, cat_name, video.strip(), "img")

    num_frames = len(os.listdir(osp.join(video_folder, cat_name, video.strip(), "img")))

    print(f"\033[91mRunning video [{vid+1}/{len(test_videos)}]: {video} with {num_frames} frames\033[0m")

    height, width = cv2.imread(osp.join(frame_folder, "00000001.jpg")).shape[:2]

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    predictions = []

    if save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_folder, f'{video_basename}.mp4'), fourcc, 30, (width, height))

    # Start processing frames
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        prompts = load_lasot_gt(osp.join(video_folder, cat_name, video.strip(), "groundtruth.txt"))

        bbox, track_label = prompts[0]
        if init_mode == 'box':
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        elif init_mode == 'mask':
            mask_img_dir = os.path.join(current_dir, "../data/sam2/lasot")
            mask_img = load_mask(osp.join(mask_img_dir, video.strip()+".png"))           
            frame_idx, object_ids, masks = predictor.add_new_mask(state, mask=mask_img, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in tqdm.tqdm(predictor.propagate_in_video(state)):
            mask_to_vis = {}
            bbox_to_vis = {}

            assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if save_to_video:

                img = cv2.imread(f'{frame_folder}/{frame_idx+1:08d}.jpg') 
                if img is None:
                    break
                
                for obj_id in mask_to_vis.keys():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask_to_vis[obj_id]] = color[(obj_id+1)%len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)
                
                for obj_id in bbox_to_vis.keys():
                    cv2.rectangle(img, (bbox_to_vis[obj_id][0], bbox_to_vis[obj_id][1]), (bbox_to_vis[obj_id][0]+bbox_to_vis[obj_id][2], bbox_to_vis[obj_id][1]+bbox_to_vis[obj_id][3]), color[(obj_id)%len(color)], 2)
                
                x1, y1, x2, y2 = prompts[frame_idx][0]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                out.write(img)

            predictions.append(bbox_to_vis)   
    os.makedirs(pred_folder, exist_ok=True)
    with open(osp.join(pred_folder, f'{video_basename}.txt'), 'w') as f:
        for pred in predictions:
            x, y, w, h = pred[0]
            f.write(f"{x},{y},{w},{h}\n")

    if save_to_video:
        out.release() 

    del predictor
    del state
    time.sleep(3)
    state = None
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    time.sleep(3)
