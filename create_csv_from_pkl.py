import glob
import os
import random
import numpy as np
import pickle
import pandas as pd
import datetime
from tqdm import tqdm
import ray
from datetime import timedelta

# TODO: I made changes to the detection algorithm in `vis_pkl_predictions.py`. 
# I need to port relevant changes over to make them ~consistent.

debug = True
out_vis_dir = '/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old'
output_csv_path = '/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/test_4M_with_NA_and_bbox.csv'
num_processes = 16
threshold_used_for_presence_of_body_part = 0.5
threshold_used_to_calc_bounding_box = 0.2
num_keypoints_to_average = 3

# https://github.com/jin-s13/COCO-WholeBody
body_parts = {
    'body': list(range(6, 18)),
    'left_foot': list(range(18, 21)),
    'right_foot': list(range(21, 24)),
    'face': list(range(1, 6)) + list(range(24, 91)),
    'left_hand': list(range(92, 113)),
    'right_hand': list(range(113, 134)),
}

# convert from 1-based to 0-based indexing
for part, indices in body_parts.items():
    body_parts[part] = [i - 1 for i in indices]

def compute_iou_of_bboxes(bbox1, bbox2):
    # Calculate the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou_value = intersection_area / union_area if union_area > 0 else 0
    return iou_value

def check_if_first_bbox_inside_second(bbox1, bbox2):
    # Check if bbox1 is inside bbox2
    x_length, y_length = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    if (bbox1[0] + 0.2 * x_length >= bbox2[0] and
        bbox1[1] + 0.2 * y_length >= bbox2[1] and
        bbox1[2] - 0.2 * x_length <= bbox2[2] and
        bbox1[3] - 0.2 * y_length <= bbox2[3]):
        return True
    return False

def compute_percent_of_bbox_inside_other_bbox(bbox1, bbox2):
    # Calculate the intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of bbox1
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    
    # Calculate the percentage of bbox1 inside bbox2
    percent_inside = intersection_area / bbox1_area if bbox1_area > 0 else 0
    return percent_inside

def get_valid_person_bbox_bool_list(person_bbox_list):
    # if there is a bounding box within another, we should discard the one inside. 
    # however, in the scenario where there are two boxes that are almost exactly the same spot and size, 
    # we should keep one of them (!)
    valid_person_bbox_bool_list = [True for _ in range(len(person_bbox_list))]
    for i, person_bbox in enumerate(person_bbox_list):
        
        # check if the bounding box is within another bounding box
        for j, other_person_bbox in enumerate(person_bbox_list):
            if i == j or other_person_bbox is None:
                continue
            
            if check_if_first_bbox_inside_second(person_bbox, other_person_bbox):
                valid_person_bbox_bool_list[i] = False
                break
    return valid_person_bbox_bool_list

def get_average_scores_per_body_part(keypoint_scores):
    # keypoint_scores = get_keypoint_scores(model, img_path)
    keypoint_scores = np.array(keypoint_scores[0])
    # keypoint_scores = (keypoint_scores[0])
    
    body_part_scores = {}
    for body_part, indices in body_parts.items():
        if body_part == 'feet':
            num_keypoints = 1
        else:
            num_keypoints = num_keypoints_to_average
        scores = keypoint_scores[list(indices)]
        # get the top N scores and average them
        scores = np.sort(scores)[::-1][:num_keypoints]
        scores = np.mean(scores)
        body_part_scores[body_part] = min(1.0, scores)
        body_part_scores[body_part] = round(float(body_part_scores[body_part]), 2)
    return body_part_scores

def get_bounding_boxes_per_body_part(keypoints, keypoint_scores):
    keypoints = np.array(keypoints[0])
    keypoint_scores = np.array(keypoint_scores[0])
    
    # keypoints: (133, 2)
    body_part_bounding_boxes = {}
    body_part_bbox_sizes = {}
    for body_part, indices in body_parts.items():
        keypoints_per_body_part = keypoints[list(indices)]
        keypoint_scores_per_body_part = keypoint_scores[list(indices)]
        
        mask = keypoint_scores_per_body_part >= threshold_used_to_calc_bounding_box
        filtered_keypoints = keypoints_per_body_part[mask]
        if filtered_keypoints.shape[0] == 0:
            body_part_bounding_boxes[body_part] = None # No keypoints above the threshold, skip this body part
            body_part_bbox_sizes[body_part] = None
            continue
        
        x_min = int(np.min(keypoints_per_body_part[:, 0]))
        y_min = int(np.min(keypoints_per_body_part[:, 1]))
        x_max = int(np.max(keypoints_per_body_part[:, 0]))
        y_max = int(np.max(keypoints_per_body_part[:, 1]))
        bounding_box = [x_min, y_min, x_max, y_max]
        body_part_bounding_boxes[body_part] = bounding_box
        
        body_part_bbox_size = abs((y_max - y_min) * (x_max - x_min))
        body_part_bbox_sizes[body_part] = body_part_bbox_size
    return body_part_bounding_boxes, body_part_bbox_sizes

def detect_if_body_part_in_image(keypoint_scores):
    average_scores_per_body_part = get_average_scores_per_body_part(keypoint_scores)

    body_in_image = int(average_scores_per_body_part['body'] >= threshold_used_for_presence_of_body_part)
    face_in_image = int(average_scores_per_body_part['face'] >= threshold_used_for_presence_of_body_part)
    left_hand_in_image = int(average_scores_per_body_part['left_hand'] >= threshold_used_for_presence_of_body_part)
    right_hand_in_image = int(average_scores_per_body_part['right_hand'] >= threshold_used_for_presence_of_body_part)
    left_foot_in_image = int(average_scores_per_body_part['left_foot'] >= threshold_used_for_presence_of_body_part)
    right_foot_in_image = int(average_scores_per_body_part['right_foot'] >= threshold_used_for_presence_of_body_part)
    
    return body_in_image, face_in_image, left_hand_in_image, right_hand_in_image, left_foot_in_image, right_foot_in_image
    
# create a pandas dataframe to store the results
def create_dataframe(vis_pkl_paths):
    df_out = pd.DataFrame(columns=['superseded_gcp_name_feb25', 'time_in_extended_iso', 
                                   'person_detected', 
                                   'person_bbox_xyxy', 
                                   'person_bbox_conf', 'person_bbox_size',
                                   'body_score', 'face_score', 'hands_score', 'feet_score', 
                                   'body_in_image', 'face_in_image', 'left_hand_in_image', 'right_hand_in_image', 'left_foot_in_image', 'right_foot_in_image',
                                   'body_bounding_box_xyxy', 'face_bounding_box_xyxy', 
                                   'left_hand_bounding_box_xyxy', 'right_hand_bounding_box_xyxy', 
                                   'left_foot_bounding_box_xyxy', 'right_foot_bounding_box_xyxy',
                                   'body_bounding_box_size', 'face_bounding_box_size', 
                                   'left_hand_bounding_box_size', 'right_hand_bounding_box_size',
                                   'left_foot_bounding_box_size', 'right_foot_bounding_box_size',
                                   ], index=None)

    for i, pkl_path in enumerate(tqdm(vis_pkl_paths, desc="Processing PKL files")):
        with open(pkl_path, 'rb') as f:
            video_hashid = os.path.basename(os.path.dirname(pkl_path)).replace('_processed', '')
            second_in_video = int(os.path.basename(pkl_path).replace('.pkl', ''))
            time_in_extended_iso = str(datetime.timedelta(seconds=second_in_video))
            
            pkl_dict = pickle.load(f)
            pose_dict = pkl_dict['pose_dict']
            
            person_bbox_list = pkl_dict['person_detection_dict']['person_bboxes']
            person_bbox_conf_list = pkl_dict['person_detection_dict']['person_confs']
            
            if len(pose_dict) == 0:
                # Create a blank entry where everything except superseded_gcp_name_feb25 and time_in_extended_iso is None
                df_out.loc[len(df_out)] = {
                    'superseded_gcp_name_feb25': video_hashid, 'time_in_extended_iso': time_in_extended_iso,
                    'person_detected': 0,
                }
                continue
            
            for i, per_person_pose_dict in pose_dict.items():
                keypoint_scores = per_person_pose_dict['keypoint_scores']
                average_scores_per_body_part = get_average_scores_per_body_part(keypoint_scores)

                body_in_image, face_in_image, left_hand_in_image, right_hand_in_image, left_foot_in_image, right_foot_in_image = detect_if_body_part_in_image(keypoint_scores)

                keypoints = per_person_pose_dict['keypoints']
                bounding_boxes_per_body_part, body_part_bbox_sizes = get_bounding_boxes_per_body_part(keypoints, keypoint_scores)

                person_bbox = [int(n) for n in person_bbox_list[i]]
                person_bbox_size = int(abs((person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])))
                person_bbox_conf = np.round(person_bbox_conf_list[i].cpu().numpy(), 2)
            
                # append to the dataframe
                df_out.loc[len(df_out)] = {
                    'superseded_gcp_name_feb25': video_hashid,
                    'time_in_extended_iso': time_in_extended_iso,
                    'person_detected': 1,
                    'person_bbox_xyxy': person_bbox, 
                    'person_bbox_conf': person_bbox_conf,
                    'person_bbox_size': person_bbox_size,
                    'body_score': average_scores_per_body_part['body'],
                    'face_score': average_scores_per_body_part['face'],
                    'hands_score': average_scores_per_body_part['hands'],
                    'feet_score': average_scores_per_body_part['feet'],
                    'body_in_image': body_in_image,
                    'face_in_image': face_in_image,
                    'left_hand_in_image': left_hand_in_image,
                    'right_hand_in_image': right_hand_in_image,
                    'left_foot_in_image': left_foot_in_image,
                    'right_foot_in_image': right_foot_in_image,
                    'body_bounding_box_xyxy': bounding_boxes_per_body_part['body'],
                    'face_bounding_box_xyxy': bounding_boxes_per_body_part['face'],
                    'left_hand_bounding_box_xyxy': bounding_boxes_per_body_part['left_hand'],
                    'right_hand_bounding_box_xyxy': bounding_boxes_per_body_part['right_hand'],
                    'feet_bounding_box_xyxy': bounding_boxes_per_body_part['feet'],
                    'body_bounding_box_size': body_part_bbox_sizes['body'],
                    'face_bounding_box_size': body_part_bbox_sizes['face'],
                    'left_hand_bounding_box_size': body_part_bbox_sizes['left_hand'],
                    'right_hand_bounding_box_size': body_part_bbox_sizes['right_hand'],
                    'feet_bounding_box_size': body_part_bbox_sizes['feet'],
                }
                
    return df_out

@ray.remote(num_gpus=0.5)
def create_dataframe_remote(vis_pkl_paths):
    return create_dataframe(vis_pkl_paths)

if __name__ == '__main__':
    vis_pkl_paths = glob.glob(os.path.join(out_vis_dir, '**/*.pkl'), recursive=True)
    random.shuffle(vis_pkl_paths)
    print('Number of PKL files:', len(vis_pkl_paths))

    if debug:
        vis_pkl_paths = vis_pkl_paths[:100]
        # debug_df = create_dataframe(vis_pkl_paths)
        # debug_df.to_csv(output_csv_path, index=False)
        # exit()

    # Split the PKL paths into chunks for parallel processing
    # Initialize Ray
    ray.init(num_cpus=num_processes)
    num_chunks = num_processes
    chunks = np.array_split(vis_pkl_paths, num_chunks)

    # Process each chunk in parallel
    dataframes = ray.get([create_dataframe_remote.remote(chunk) for chunk in chunks])


    # Combine all dataframes into a single dataframe
    final_dataframe = pd.concat(dataframes, ignore_index=True)

    # Save the combined dataframe to a CSV file
    final_dataframe.to_csv(output_csv_path, index=False)

    # Print some statistics
    print(final_dataframe.round(2).describe())
