import argparse
import glob
import os
import random
import numpy as np

import ray

from tqdm import tqdm
import time

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmcv.image import imread
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from ultralytics import YOLO

"""
    # --input_frames_txt_file /ccn2/dataset/babyview/outputs_20250312/1000_random_frames.txt \
conda activate babyview-pose
cd /ccn2/u/khaiaw/Code/babyview-pose/mmpose/

python run_model.py \
    --input_dir /ccn2/dataset/babyview/outputs_20250312/sampled_frames/ \
    --output_dir /ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_test_discard \
    --debug
"""

def get_args():
    parser = argparse.ArgumentParser(description='Process videos to the desired fps, resolution, rotation.')
    parser.add_argument('--input_frames_txt_file', type=str, help='Path to list of input frames')
    parser.add_argument('--input_dir', type=str, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--num_processes', type=int, default=32, help='Number of processes to run in parallel')
    return parser.parse_args()

def get_person_detection_model():
    model = YOLO("yolo12x.pt")
    return model
    
def get_pose_model():
    # config = 'downloads/rtmw-x_8xb320-270e_cocktail14-384x288.py'
    # checkpoint = 'downloads/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
    config = 'downloads/rtmw-x_8xb704-270e_cocktail14-256x192.py'
    checkpoint = 'downloads/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth'
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    device = 'cuda'
    model = init_model(config, checkpoint, device=device, cfg_options=cfg_options)
    return model

def save_result_metadata_pickle(batch_results, person_detection_dict, output_path):
    import pickle
    
    pose_dict = {}
    if batch_results is not None:
        for i, batch_result in enumerate(batch_results):
            per_person_pose_dict = {}
            for k, v in batch_result.pred_instances.items():
                if k not in ['keypoints', 'keypoint_scores']:
                    continue
                per_person_pose_dict[k] = v.astype(np.float16)
            pose_dict[i] = per_person_pose_dict
    
    # Write to file
    pkl_dict = {}
    pkl_dict['person_detection_dict'] = person_detection_dict
    pkl_dict['pose_dict'] = pose_dict
    with open(output_path, 'wb') as f:
        pickle.dump(pkl_dict, f)

class VisualizerArgs:
    def __init__(self):
        self.radius = 3
        self.alpha = 0.8
        self.thickness = 1
        self.kpt_thr = 0.3
        self.draw_heatmap = True
        self.show_kpt_idx = False
        self.skeleton_style = 'mmpose'
        self.show = False

@ray.remote(num_gpus=0.25)
def run_on_images_remote(args, img_paths):
    run_on_images(args, img_paths)
    
def run_on_images(args, img_paths):
    visualizer_args = VisualizerArgs()
    person_detection_model = get_person_detection_model()
    pose_model = get_pose_model()
    
    pose_model.cfg.visualizer.radius = visualizer_args.radius
    pose_model.cfg.visualizer.alpha = visualizer_args.alpha
    pose_model.cfg.visualizer.line_width = visualizer_args.thickness

    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_model.dataset_meta, skeleton_style=visualizer_args.skeleton_style)
    
    for i, img_path in enumerate(tqdm(img_paths, desc="Processing images")):
        dir_and_filename = '/'.join(img_path.split('/')[-2:])
        out_vis_path = os.path.join(args.output_dir, dir_and_filename)
        out_pkl_path = out_vis_path.replace('.jpg', '.pkl')
        os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
        
        if os.path.exists(out_pkl_path):
            continue
        
        person_detection_result = person_detection_model(img_path, classes=[0])[0]
        person_bboxes = person_detection_result.boxes.xyxy 
        person_bboxes = person_bboxes.cpu().numpy() # [N, 4]
        person_confs = person_detection_result.boxes.conf
        person_detection_dict = {
            'person_bboxes': person_bboxes,
            'person_confs': person_confs
        }
        
        # === Save metadata into pickle ===
        if len(person_bboxes) == 0:
            save_result_metadata_pickle(None, person_detection_dict, out_pkl_path)
        else:
            batch_results = inference_topdown(pose_model, img_path, person_bboxes, bbox_format='xyxy')
            save_result_metadata_pickle(batch_results, person_detection_dict, out_pkl_path)

        # === Visualize every N samples ===
        if i % 1000 == 0:
            if len(person_bboxes) == 0:
                os.system(f'cp {img_path} {out_vis_path}') # simply copy the image, skip the next step as that is costly
            else:
                results = merge_data_samples(batch_results)
                img = imread(img_path, channel_order='rgb')
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=results,
                    draw_gt=False,
                    draw_bbox=True,
                    kpt_thr=visualizer_args.kpt_thr,
                    draw_heatmap=visualizer_args.draw_heatmap,
                    show_kpt_idx=visualizer_args.show_kpt_idx,
                    skeleton_style=visualizer_args.skeleton_style,
                    show=visualizer_args.show,
                    out_file=out_vis_path)

def main(args):
    # with open('/ccn2/dataset/babyview/outputs_20250312/10000_random_frames.txt', 'r') as f:
    #     img_path_list = f.readlines()
    #     img_path_list = [x.strip() for x in img_path_list]
    
    if args.input_dir:
        img_path_list = glob.glob(os.path.join(args.input_dir, '**/*.jpg'), recursive=True)
        
    if args.input_frames_txt_file:
        with open(args.input_frames_txt_file, 'r') as f:
            img_path_list = f.readlines()
            img_path_list = [x.strip() for x in img_path_list]
    
    print('img_path_list:', len(img_path_list))
    
    # debugging
    if args.debug:
        img_path_list = img_path_list[:16]
        img_path_list = ['/ccn2/u/khaiaw/Code/babyview-pose/mmpose/tests/data/coco/000000196141.jpg']
        run_on_images(args, img_path_list)
        return
    
    # split into chunks of arg.num_processes
    ray.init(_temp_dir="/ccn2/u/khaiaw/ray_tmp")
    chunk_size = max(1, len(img_path_list) // args.num_processes)
    img_path_chunks = [img_path_list[i:i + chunk_size] for i in range(0, len(img_path_list), chunk_size)]
    futures = [run_on_images_remote.remote(args, img_paths) for img_paths in img_path_chunks]
    ray.get(futures)

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)

