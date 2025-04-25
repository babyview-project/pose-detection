import glob
import os
import pickle
import random
import numpy as np

from create_csv_from_pkl import check_if_first_bbox_inside_second, compute_iou_of_bboxes, compute_percent_of_bbox_inside_other_bbox, detect_if_body_part_in_image, get_average_scores_per_body_part, get_bounding_boxes_per_body_part, get_valid_person_bbox_bool_list

from tqdm import tqdm
import time
import cv2
import shutil

debug = False
seed = 9999
specified_frames_txt_file = '/ccn2/dataset/babyview/outputs_20250312/100_sampled_frames.txt'
with open(specified_frames_txt_file, 'r') as f:
    frames_list = [line.strip() for line in f if line.strip()]

pkl_dir = '/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/'
frames_dir = '/ccn2/dataset/babyview/outputs_20250312/sampled_frames/'
out_vis_dir = './vis_pkl_predictions/'
shutil.rmtree(out_vis_dir, ignore_errors=True)

body_part_colors = {
    'person': (255, 255, 255),
    'face': (255, 0, 0),
    'left_hand': (0, 255, 0),   # Bright green for left hand
    'right_hand': (0, 180, 0),  # Slightly darker green for right hand
    'body': (0, 0, 255),
    'left_foot': (255, 255, 0),
    'right_foot': (180, 180, 0),
}

pkl_paths = glob.glob(os.path.join(pkl_dir, '**/*.pkl'), recursive=True)
random.seed(seed)  # For reproducibility
random.shuffle(pkl_paths)

if debug:
    pkl_paths = pkl_paths[:100]

def draw_bbox(img, bbox, color, label, body_part_in_image):
    if not body_part_in_image:
        return
    if bbox is not None:
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, pt1, pt2, color, 2)
        # cv2.putText(img, label, (pt1[0], max(pt1[1] - 10, 0)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def add_legend(img):
    # Create a legend image at the top-left corner
    legend = np.zeros((160, 80, 3), dtype=np.uint8)
    legend[:] = (0, 0, 0)
    cv2.putText(legend, 'Person', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['person'], 2)
    cv2.putText(legend, 'Face', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['face'], 2)
    cv2.putText(legend, 'L-Hand', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['left_hand'], 2)
    cv2.putText(legend, 'R-Hand', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['right_hand'], 2)
    cv2.putText(legend, 'Body', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['body'], 2)
    cv2.putText(legend, 'L-Foot', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['left_foot'], 2)
    cv2.putText(legend, 'R-Foot', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_part_colors['right_foot'], 2)
    
    # Overlay the legend onto the image with opacity
    h, w = legend.shape[:2]
    alpha = 0.4  # opacity value between 0.0 and 1.0
    roi = img[0:h, 0:w].copy()
    blended = cv2.addWeighted(legend, alpha, roi, 1 - alpha, 0)
    img[0:h, 0:w] = blended
    

for frame_path in tqdm(frames_list):
    pkl_path = frame_path.replace(frames_dir, pkl_dir).replace('.jpg', '.pkl')
    
    frame_orig = cv2.imread(frame_path)
    vis_frame = frame_orig.copy()
    
    pkl_dict = pickle.load(open(pkl_path, 'rb'))
    pose_dict = pkl_dict['pose_dict']
    person_bbox_list = pkl_dict['person_detection_dict']['person_bboxes']
    
    valid_person_bbox_bool_list = get_valid_person_bbox_bool_list(person_bbox_list)
    
    for i, per_person_pose_dict in pose_dict.items():
        if not valid_person_bbox_bool_list[i]:
            continue
        keypoint_scores = per_person_pose_dict['keypoint_scores']
        keypoints = per_person_pose_dict['keypoints']
        
        body_in_image, face_in_image, left_hand_in_image, right_hand_in_image, left_foot_in_image, right_foot_in_image = detect_if_body_part_in_image(keypoint_scores)
        
        average_scores_per_body_part = get_average_scores_per_body_part(keypoint_scores)
        
        bounding_boxes_per_body_part, _ = get_bounding_boxes_per_body_part(keypoints, keypoint_scores)
        face_bounding_box_xyxy = bounding_boxes_per_body_part['face']
        left_hand_bounding_box_xyxy = bounding_boxes_per_body_part['left_hand']
        right_hand_bounding_box_xyxy = bounding_boxes_per_body_part['right_hand']
        body_bounding_box_xyxy = bounding_boxes_per_body_part['body']
        left_foot_bounding_box_xyxy = bounding_boxes_per_body_part['left_foot']
        right_foot_bounding_box_xyxy = bounding_boxes_per_body_part['right_foot']
        
        if left_hand_in_image and right_hand_in_image:
            iou_hands = compute_iou_of_bboxes(left_hand_bounding_box_xyxy, right_hand_bounding_box_xyxy)
            percent_bbox_left_hand_inside_right_hand = compute_percent_of_bbox_inside_other_bbox(left_hand_bounding_box_xyxy, right_hand_bounding_box_xyxy)
            percent_bbox_right_hand_inside_left_hand = compute_percent_of_bbox_inside_other_bbox(right_hand_bounding_box_xyxy, left_hand_bounding_box_xyxy)
            is_one_bbox_inside_another = check_if_first_bbox_inside_second(left_hand_bounding_box_xyxy, right_hand_bounding_box_xyxy) \
                                    or check_if_first_bbox_inside_second(right_hand_bounding_box_xyxy, left_hand_bounding_box_xyxy)
            if iou_hands > 0.3 or is_one_bbox_inside_another or percent_bbox_left_hand_inside_right_hand > 0.5 or percent_bbox_right_hand_inside_left_hand > 0.5:
                left_hand_in_image = average_scores_per_body_part['left_hand'] >= average_scores_per_body_part['right_hand']
                right_hand_in_image = average_scores_per_body_part['right_hand'] > average_scores_per_body_part['left_hand']
        
        draw_bbox(vis_frame, face_bounding_box_xyxy, body_part_colors['face'], 'Face', face_in_image)
        draw_bbox(vis_frame, left_hand_bounding_box_xyxy, body_part_colors['left_hand'], 'Hands', left_hand_in_image)
        draw_bbox(vis_frame, right_hand_bounding_box_xyxy, body_part_colors['right_hand'], 'Hands', right_hand_in_image)
        # draw_bbox(vis_frame, body_bounding_box_xyxy, body_part_colors['body'], 'Body', body_in_image)
        draw_bbox(vis_frame, left_foot_bounding_box_xyxy, body_part_colors['left_foot'], 'Feet', left_foot_in_image)
        draw_bbox(vis_frame, right_foot_bounding_box_xyxy, body_part_colors['right_foot'], 'Feet', right_foot_in_image)
        
        # Person bbpx
        person_bbox_xyxy = person_bbox_list[i]
        draw_bbox(vis_frame, person_bbox_xyxy, body_part_colors['person'], 'Person', True)

        add_legend(vis_frame)
        
    # Save the visualized frame
    out_vis_path = frame_path.replace(frames_dir, out_vis_dir)
    split_path = out_vis_path.rsplit(os.sep, 1)
    if len(split_path) == 2:
        out_vis_path = split_path[0] + '_' + split_path[1]
    os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
    cv2.imwrite(out_vis_path, vis_frame)
    
out_vis_dir_with_seed = os.path.join('./viz_pkl_predictions', f'seed_{seed}')
if os.path.exists(out_vis_dir_with_seed):
    shutil.rmtree(out_vis_dir_with_seed)
shutil.move(out_vis_dir, out_vis_dir_with_seed)