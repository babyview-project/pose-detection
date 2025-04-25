# pose-detection

Background
- Goal: Generate bounding box detections for hands, faces, and bodies
- We first use a person detection model (e.g., YOLO) to detect person bounding boxes, and then use a pose detection model (e.g., RTMW pose) (labeling keypoints of the model) to detect the faces/hands/bodies/feet within each of those bounding boxes.
- We use this approach rather than directly using an object detection model because we cannot find any object detection models trained on hands vs faces vs bodies (and open-world detection models do not work well on these categories)
- Faces, hands, bodies are something neuroscience people care about a lot and think are highly important categories (see research on identifying face and body areas in the brain)
- An alternative approach that might work well is using a separate hand detection model and face detection model. (if we do not care about "bodies")

Notes
- This repo builds upon https://github.com/open-mmlab/mmpose
- The main files used for BabyView stuff are:
  - `run_model.py`: This runs the person detection model, followed by the pose model, which saves results as pickle files
  - `create_csv_from_pkl.py`: This creates a csv from the pickle files

Setup 
- `conda create -n babyview-pose python=3.12`
- Install mmpose: https://mmpose.readthedocs.io/en/latest/installation.html (mmpose can sometimes be painful to setup if you run into error messages but there is no easy other way to do this)
- Install YOLO: https://docs.ultralytics.com/quickstart/ (this was easy for me)
- I provided a `requirements.txt` file but I am not sure if it works to just install with `pip install -r requirements.txt`

Download models/configs for mmpose
- Download the desired configs and models used in `run_model.py` from mmpose github.
    - Place the configs and models in `downloads/` directory in this repo
    - This will differ depending on what model you wish to run
    - E.g., RTMW mmpose configs and models can be found in the hyperlinks of this [page](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md)
