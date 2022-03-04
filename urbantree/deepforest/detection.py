import os
import time
import math
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import inspect
import warnings
import logging
import rasterio
import cv2
import scipy.ndimage
import skimage.morphology
from concurrent.futures import ThreadPoolExecutor
import dask
import shapely
import geopandas as gpd
from deepforest import main, preprocess, predict, visualize
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.ops import nms

def distance(points):
  """
  calculate the distance of two points

  Parameters
  ----------
  points : list
    a list of two points. E.g. `[[x1, y1], [x2, y2]]`

  Returns
  -------
  float
    the distance of two points
  """
  p1, p2 = points
  return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[1]-p2[1],2))

def calc_rectangle_bbox(points, img_h, img_w):
  """
  calculate bbox from a rectangle.

  Parameters
  ----------
  points : list
    a list of two points. E.g. `[[x1, y1], [x2, y2]]`
  img_h : int
    maximal image height
  img_w : int
    maximal image width

  Returns
  -------
  dict
    corresponding bbox. I.e. `{ 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax }`
  """
  lt, rb = points
  xmin, ymin = lt
  xmax, ymax = rb
  xmin = min(max(0, xmin), img_w)
  xmax = min(max(0, xmax), img_w)
  ymin = min(max(0, ymin), img_h)
  ymax = min(max(0, ymax), img_h)
  return { 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax }

def calc_circle_bbox(points, img_h, img_w):
  """
  calculate bbox from a circle.

  Parameters
  ----------
  points : list
    a list of two points. E.g. `[[x1, y1], [x2, y2]]`
  img_h : int
    maximal image height
  img_w : int
    maximal image width

  Returns
  -------
  dict
    corresponding bbox. I.e. `{ 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax }`

  """
  center = points[0]
  dist = distance(points)
  xmin = center[0] - dist
  xmax = center[0] + dist
  ymin = center[1] - dist
  ymax = center[1] + dist
  xmin = min(max(0, xmin), img_w)
  xmax = min(max(0, xmax), img_w)
  ymin = min(max(0, ymin), img_h)
  ymax = min(max(0, ymax), img_h)
  return { 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax }

def generate_response(dataset_train_dir, dataset_response_dir,
                      model_trainer_min_bbox_size,
                      model_trainer_min_bbox_ratio,
                      model_trainer_validation_ratio,
                      model_trainer_patch_sizes=[],
                      model_trainer_patch_overlap_size=32,
                      **ignored):
  """
  generate response of training dataset with labelme annotation results.

  Parameters
  ----------
  dataset_train_dir : str
    the path to input training dataset folder with labelme annotation results in json
  dataset_response_dir : str
    the path to output training response folder for further training with torch vision
  model_trainer_min_bbox_size : int
    minimal size of object bbox which should be trained
  model_trainer_min_bbox_ratio : float
    minimal ratio (short_side/long_side) of object bbox
  model_trainer_validation_ratio : float
    train/validation split ratio
  model_trainer_patch_sizes : list of int
    a list of patch sizes for training images
  model_trainer_patch_overlap_size : int
    overlapping size of cropped training images with the given patch size
  """

  TRAINING_IMG_DIR = Path(dataset_train_dir)

  def glob_json(path):
    return list(path.glob('*.json'))

  # TRAINING_IMG_PATH can be a list of directories
  LABELME_RESULTS = [glob_json(TRAINING_IMG_DIR)] \
                    if not isinstance(TRAINING_IMG_DIR, list) \
                    else list(map(glob_json, TRAINING_IMG_DIR))

  # output folder of response (annotation csv fot torch vision)
  RESPONSE_DEEPFOREST_DIR = Path(dataset_response_dir)

  PATCH_SIZES        = model_trainer_patch_sizes
  PATCH_OVERLAP_SIZE = model_trainer_patch_overlap_size
  VALIDATION_RATIO   = model_trainer_validation_ratio
  TARGET_MIN_SIZE    = model_trainer_min_bbox_size
  TARGET_MIN_RATIO   = model_trainer_min_bbox_ratio

  assert TARGET_MIN_RATIO <= 1 and TARGET_MIN_RATIO > 0, "0 < model_trainer_min_bbox_ratio <= 1"
  #############################################################

  # need to start from scratch
  if RESPONSE_DEEPFOREST_DIR.exists() and os.listdir(RESPONSE_DEEPFOREST_DIR):
    raise RuntimeError("Directory is not empty: " + str(RESPONSE_DEEPFOREST_DIR))
  os.makedirs(RESPONSE_DEEPFOREST_DIR, exist_ok=True)

  def bbox_size(bbox):
    return (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin'])
  def bbox_ratio(bbox):
    return (bbox['xmax'] - bbox['xmin']) / (bbox['ymax'] - bbox['ymin'])

  print("Processing source labelme annotation:", TRAINING_IMG_DIR)
  all_df = []
  for dir_id, label_jsons in enumerate(LABELME_RESULTS):
    for lbme in label_jsons:
      # labelme json file
      lbme_path = Path(lbme)
      image_json = None
      with open(lbme) as f:
        image_json = json.load(f)

      img_h = image_json['imageHeight']
      img_w = image_json['imageWidth']
      img_dir       = lbme.parent
      src_img_name  = image_json['imagePath']
      dest_img_name = str(dir_id) + '-' + src_img_name

      # copy training data to response folder
      shutil.copy(img_dir.joinpath(src_img_name),
                  RESPONSE_DEEPFOREST_DIR.joinpath(dest_img_name))

      rows_list = []
      # process geometries
      for shape in image_json['shapes']:
        shape_type = shape['shape_type']
        shape_label = shape['label']
        shape_points = shape['points']

        bbox = None
        if shape_type == 'circle':
          bbox = calc_circle_bbox(points=shape_points, img_h=img_h, img_w=img_w)
        elif shape_type == 'rectangle':
          bbox = calc_rectangle_bbox(points=shape_points, img_h=img_h, img_w=img_w)
        else:
          raise "FIXME"

        if (bbox_size(bbox) >= TARGET_MIN_SIZE) and \
            ((bbox_ratio(bbox) >= TARGET_MIN_RATIO) and (bbox_ratio(bbox) <= 1.0/TARGET_MIN_RATIO)):
          row = { 'image_path': dest_img_name, **bbox, 'label': shape_label }
          rows_list.append(row)

      df = pd.DataFrame(rows_list)
      dest = RESPONSE_DEEPFOREST_DIR / (str(dir_id) + '-' + lbme_path.stem + '.csv')
      df.to_csv(dest, index=False)
      all_df.append(df)

  if len(all_df):
    print("output response dir:", RESPONSE_DEEPFOREST_DIR)
    # collect all annotation
    combined_annotations = pd.concat(all_df, ignore_index=True)
    combined_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "combined.all.csv_", index=False)
    # random split to train/valid set
    image_paths = combined_annotations.image_path.unique()
    valid_paths = np.random.choice(image_paths, int(len(image_paths)*VALIDATION_RATIO))
    valid_annotations = combined_annotations.loc[combined_annotations.image_path.isin(valid_paths)]
    train_annotations = combined_annotations.loc[~combined_annotations.image_path.isin(valid_paths)]
    train_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "combined.train.csv_",  index=False)
    valid_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "combined.valid.csv_" , index=False)

  # cropping
  print("Cropping response dataset:", RESPONSE_DEEPFOREST_DIR)
  all_df = []
  for _, PATCH_SIZE in enumerate(PATCH_SIZES):
    for annotation_file in list(RESPONSE_DEEPFOREST_DIR.glob("*.csv")):
      path_to_raster = RESPONSE_DEEPFOREST_DIR.joinpath(annotation_file.stem + ".tiff")
      CROP_BASE_DIR = RESPONSE_DEEPFOREST_DIR / "crop" / str(PATCH_SIZE)
      df = preprocess.split_raster(annotations_file=annotation_file,
                                path_to_raster=path_to_raster,
                                base_dir=CROP_BASE_DIR,
                                patch_size=PATCH_SIZE,
                                patch_overlap=1.0 * PATCH_OVERLAP_SIZE / PATCH_SIZE)
      df['image_path'] = str(PATCH_SIZE) + '/' + df['image_path']
      all_df.append(df)

  # collect all annotation
  if len(all_df):
    print("output response dir:", (RESPONSE_DEEPFOREST_DIR / "crop"))
    combined_annotations = pd.concat(all_df, ignore_index=True)
    combined_annotations = combined_annotations[
                            (bbox_size(combined_annotations) >= TARGET_MIN_SIZE) &
                            ((bbox_ratio(combined_annotations) >= TARGET_MIN_RATIO) &
                             (bbox_ratio(combined_annotations) <= 1.0/TARGET_MIN_RATIO))]
    combined_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "crop" / "combined.all.csv_", index=False)
    # split to train/valid
    image_paths = combined_annotations.image_path.unique()
    valid_paths = np.random.choice(image_paths, int(len(image_paths)*VALIDATION_RATIO))
    valid_annotations = combined_annotations.loc[combined_annotations.image_path.isin(valid_paths)]
    train_annotations = combined_annotations.loc[~combined_annotations.image_path.isin(valid_paths)]
    train_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "crop" / "combined.train.csv_",  index=False)
    valid_annotations.to_csv(RESPONSE_DEEPFOREST_DIR / "crop" / "combined.valid.csv_" , index=False)

def train_model(model_params_config,
                model_trainer_logging_dir,
                pretrained_model_path=None,
                save_top_k=3,
                **ignored):
  """
  model training

  Parameters
  ----------
  model_params_config : dict
    model parameter for `DeepForest`
  model_trainer_logging_dir : str
    the path to output folder of training logs and checkpoints
  pretrained_model_path : str
    the path to any pre-trained model
  """
  MODEL_CONFIG = model_params_config

  MODEL_LOG_DIR            = Path(model_trainer_logging_dir)
  MODEL_LOG_CHECKPOINT_DIR = MODEL_LOG_DIR.joinpath('ckpt')
  MODEL_LOGGING = {
    'logger': TensorBoardLogger(save_dir=MODEL_LOG_DIR),
    'log_every_n_steps': 50
  }

  PRETRAINED_MODEL_PATH = pretrained_model_path

  #############################################################

  print("Trainer setting:", MODEL_CONFIG)
  print("TensorBoard logs will be saved in", MODEL_LOG_DIR)
  print("Checkpoints will be saved in", MODEL_LOG_CHECKPOINT_DIR)

  model = main.deepforest()
  model.use_release()

  if PRETRAINED_MODEL_PATH is not None:
    print("Loading pre-trained model:", PRETRAINED_MODEL_PATH)
    model = main.deepforest.load_from_checkpoint(checkpoint_path=PRETRAINED_MODEL_PATH)

  model.config.update(MODEL_CONFIG) # TODO: better with careful nested update

  ts = str(int(time.time()))
  box_recall_callback = ModelCheckpoint(dirpath=MODEL_LOG_CHECKPOINT_DIR,
                                        monitor='box_recall', # or 'box_precision'
                                        mode="max",
                                        save_top_k=save_top_k,
                                        filename=ts + "_m_box_recall-{epoch:02d}-{box_recall:.2f}-{box_precision:.2f}")
  box_precision_callback = ModelCheckpoint(dirpath=MODEL_LOG_CHECKPOINT_DIR,
                                           monitor='box_precision', # or 'box_precision'
                                           mode="max",
                                           save_top_k=save_top_k,
                                           filename=ts + "_m_box_precision-{epoch:02d}-{box_recall:.2f}-{box_precision:.2f}")
  model.create_trainer(**MODEL_LOGGING,
                       callbacks=[box_recall_callback, box_precision_callback])
  model.trainer.fit(model)

  #  results = model.evaluate(csv_file=model.config['validation']['csv_file'],
  #                          root_dir=model.config['validation']['root_dir'],
  #                          iou_threshold=model.config['validation']['iou_threshold'])
  #  print("precision:", results['box_precision'])
  #  print("recall:", results['box_recall'])

def remove_nested_bbox(df):
  """
  remove nested bbox

  Parameters
  ----------
  df : DataFrame
    inference result with bounding boxes, labels and scores
  """
  df['__id'] = range(0, df.shape[0]) # make an extra 'id'
  df['__geometry'] = df.apply(
      lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
  ddf = gpd.GeoDataFrame(df, geometry='__geometry')
  ddf['__area'] = ddf.__geometry.area
  ddf.sort_values('__area', ascending=False, inplace=True)

  nested_id = set()
  for iloc in range(ddf.shape[0]):
      if ddf.iloc[iloc]['__id'] in nested_id:
          continue
      bbox = ddf.iloc[iloc]
      test = ddf.iloc[iloc+1:]
      nested = test[~test['__id'].isin(nested_id) &
                    test['__geometry'].within(bbox['__geometry'])]
      nested_id.update(list(nested['__id']))

  df = df[~df['__id'].isin(nested_id)]
  del df['__id']
  del df['__geometry']
  del df['__area']

  return df

def run_nms(df, use_soft_nms=False, iou_threshold=0.15,
            sigma=0.5, score_threshold=0.001):
  """
  Non-maximum suppression

  Parameters
  ----------
  df : DataFrame
    inference result with bounding boxes, labels and scores
  use_soft_nms : bool
    use `nms` or `soft nms`
  iou_threshold : float
    IoU threshold for `nms`
  sigma : float
    sigma for `soft nms`
  score_threshold : float
    score threshold for `soft nms`
  """
  boxes = torch.tensor(df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
  scores = torch.tensor(df.score.values, dtype=torch.float32)

  if use_soft_nms:
    bbox_left_idx = predict.soft_nms(boxes=boxes,
                                     scores=scores,
                                     sigma=sigma,
                                     thresh=score_threshold).numpy()
  else:
    bbox_left_idx = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold).numpy()
  return df.iloc[bbox_left_idx]

def infer_image(model, img_path, out_dir,
                param_model_inference, prefer_model_params_config,
                continue_mode=True):
  """
  detect objects from a image.
  The object detection would be performed on different patch sizes and
  Note: no extra nms is run on aggregated results from different patch sizes.

  Parameters
  ----------
  model : object
    trained DeepForest model object
  img_path : str
    image file path
  out_dir : str
    output result folder path
  param_model_inference : dict
    inference parameter
  continue_mode : bool
    continue without working on already processed images
  prefer_model_params_config : bool
    use `nms_thresh` and `score_thresh` from model_params_config instead of `iou_threshold` and `score_thresh` from model_inference_config.
    This can be useful if a separated postprocess will take place to perform some fine tuning.
  """
  # output path
  out_path = out_dir / (img_path.stem + ".pkl")
  if out_path.exists() and continue_mode:
    logging.debug("PASSED: {out_path}".format(out_path=out_path))
    return
  # load image
  image = np.array(Image.open(img_path).convert("RGB")).astype("uint8")
  # predict (inference with multiple scales/patches)
  predicted_boxes = []
  for model_inf in param_model_inference['patch']:
    stored_config = model.config.copy()
    # update model's config before inference if necessary
    if prefer_model_params_config:
      # Note: we keep model.config['score_thresh'] with default low value to get as much as possible results
      if model.config['score_thresh'] > model_inf['score_thresh']:
        warnings.warn("Model inference parameter for 'patch_size={p}' ".format(p=model_inf['patch_size']) +
                      "has a smaller 'score_thresh={i}' ".format(i=model_inf['score_thresh']) +
                      "than default model.config['score_thresh']={i}".format(i=model.config['score_thresh']))
      if model.config['nms_thresh'] < model_inf['iou_threshold']:
        warnings.warn("Model inference parameter for 'patch_size={p}' ".format(p=model_inf['patch_size']) +
                      "has a larger 'iou_threshold={i}' ".format(i=model_inf['iou_threshold']) +
                      "than default model.config['nms_thresh']={i}".format(i=model.config['nms_thresh']))
    else:
      model.config['score_thresh'] = model_inf['score_thresh']
      model.config['nms_thresh']   = model_inf['iou_threshold']
    # predict_tile returns bboxes
    predict_tile_param = {k: v for k, v in model_inf.items() if k in inspect.getfullargspec(model.predict_tile).args}
    predicted = model.predict_tile(image=image, return_plot=False, **predict_tile_param)
    if predicted is not None:
      # add extra meta info
      predicted["patch_size"] = model_inf["patch_size"]
      predicted_boxes.append(predicted)
    # restore model's config
    model.config = stored_config

  if len(predicted_boxes) > 0:
    df = pd.concat(predicted_boxes, ignore_index=True)
    df.to_pickle(out_path)
    logging.debug("PROCESSED: {out_path}".format(out_path=out_path))
  else:
    logging.debug("SKIPPED: {out_path}".format(out_path=out_path))


def infer_images(model_path, model_params_config, model_inference_config,
                 dataset_inference_dir,
                 dataset_img_dir=None,
                 dataset_img_list=None,
                 dataset_img_pattern="*.tiff",
                 continue_mode=True, use_gpu=True, prefer_model_params_config=True, **ignored):
  """
  detect tree objects from dataset.
  The object detection would be performed on different patch sizes and
  Note: no extra nms is run on aggregated results from different patch sizes.

  Parameters
  ----------
  model_path : str
    the path to the trained model
  model_params_config : dict
    model parameter for `DeepForest`
  model_inference_config : dict
    inference parameter
  dataset_img_dir : str
    the path to image dataset folder to infer
  dataset_img_list : list
    a list of image paths as input dataset to infer
  dataset_inference_dir : str
    the path to the output folder of inference result.
    Resulting bbox in *.pkl will be stored in subfolder `b/` of the output folder
  continue_mode : bool
    continue without working on already processed images
  use_gpu : bool
    use GPU if possible
  prefer_model_params_config : bool
    use `nms_thresh` and `score_thresh` from model_params_config instead of `iou_threshold` and `score_thresh` from model_inference_config
    This can be useful if a separated postprocess will take place to perform some fine tuning.
  """
  CONTINUE_MODE      = continue_mode
  USE_GPU            = use_gpu

  DATASET_IMG_DIR    = dataset_img_dir
  DATASET_IMG_LIST   = dataset_img_list
  assert dataset_img_dir is not None or dataset_img_list is not None, "Either `dataset_img_dir` or `dataset_img_list` must be defined"
  INFERENCE_RESULT_DIR = Path(dataset_inference_dir).joinpath('b')

  MODEL_CHECKPOINT   = model_path

  MODEL_CONFIG       = model_params_config
  MODEL_INFERENCE    = model_inference_config

  #############################################################

  print("Loading pretrained mode:", MODEL_CHECKPOINT)
  print("Inference result:", INFERENCE_RESULT_DIR)

  os.makedirs(INFERENCE_RESULT_DIR, exist_ok=True)

  model = main.deepforest()
  model.use_release()
  model = main.deepforest.load_from_checkpoint(checkpoint_path=MODEL_CHECKPOINT)
  model.config.update(MODEL_CONFIG)

  if USE_GPU and torch.cuda.is_available():
    model.to("cuda")

  IMAGE_LIST = DATASET_IMG_LIST if DATASET_IMG_LIST is not None else Path(DATASET_IMG_DIR).glob(dataset_img_pattern)
  for src in IMAGE_LIST:
    infer_image(model=model, img_path=src,
                    out_dir=INFERENCE_RESULT_DIR, param_model_inference=MODEL_INFERENCE,
                    prefer_model_params_config=prefer_model_params_config, continue_mode=CONTINUE_MODE)

def filter_bbox(df, model_inference_config):
  """
  filter bounding boxes according to size and scores.

  Parameters
  ----------
  df : DataFrame
    dataframe of bounding boxes
  model_inference_config : dict
    model inference config object

  Returns
  -------
  DataFrame
    filtered bounding boxes
  """
  # load bbox: filter after score_thresh, keep small trees only with very high scores
  filtered_df = []
  for model_inf in model_inference_config['patch']:
    filtered_df.append(df[(df.patch_size == model_inf['patch_size']) &
                          (df.score >= model_inf['score_thresh'])])
  df = pd.concat(filtered_df, ignore_index=True)
  df = df[(df.score >= model_inference_config['confident_min_score']) |
          ((df.xmax-df.xmin)*(df.ymax-df.ymin) >= model_inference_config['confident_min_bbox_size'])]
  return df

def draw_bbox(src_image_path, output_image_path, src_bbox_path=None, src_bbox_df=None):
  """
  draw bounding boxes on images

  Parameters
  ----------
  src_img_path : str
    the path to the source image
  output_image_path : str
    the path to the output image
  src_bbox_path : str
    the path to the bounding box pickle files
  src_bbox_df : str
    dataframe of bounding boxes
  """
  with rasterio.open(src_image_path) as src:
    image_crs = src.crs
    image_transform = src.transform
    image_height = src.height
    image_width = src.width

  image = np.array(Image.open(src_image_path).convert("RGB")).astype("uint8")
  boxes = src_bbox_df if src_bbox_df is not None else pd.read_pickle(src_bbox_path)
  image = image[:,:,::-1] # RGB => BGR
  image = visualize.plot_predictions(image, boxes)
  image = image[:,:,::-1] # BGR => RGB

  DATASET_IMG_META = {
    'driver': 'GTiff',
    'dtype':  'uint8',
    'width':  image_width,
    'height': image_height,
    'count':  3,
    'crs':    image_crs,
    'transform': image_transform
  }

  Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
  with rasterio.open(output_image_path, 'w', **DATASET_IMG_META) as dst:
    dst.write(np.moveaxis(image, -1, 0)) # (H, W, C) -> (C, H, W)

def postprocess_render_image(src_img_path, src_bbox_dir,
                             output_img_dir, model_inference_config,
                             continue_mode):
  """
  Render tree canopy as raster images from detected bboxes.

  Parameters
  ----------
  src_img_path : str
    the path to the source image
  src_bbox_dir : str
    the path to the folder containing bbox result (*.pkl)
  output_img_dir : str
    the path to the output folder of rendered results
  model_inference_config : dict
    inference parameter
  continue_mode : bool
    continue without working on already processed images
  """
  img_name = src_img_path.stem
  src_bbox_path = src_bbox_dir.joinpath(img_name + ".pkl")
  out_img_path = output_img_dir.joinpath(img_name + ".tiff")

  if (out_img_path.exists() and continue_mode) or (not src_bbox_path.exists()):
    logging.debug("PASSED: {p}".format(p=out_img_path))
    return

  # load image and meta
  image_shape, image_crs, image_transform = None, None, None
  with rasterio.open(src_img_path) as src:
    image_shape = src.shape
    image_crs = src.crs
    image_transform = src.transform
  # load bbox: filter after score_thresh, keep small trees only with very high scores
  df = pd.read_pickle(src_bbox_path)
  df = filter_bbox(df=df, model_inference_config=model_inference_config)
  # draw raster result
  out_img = np.zeros(image_shape, np.uint8)
  for _, box in df.iterrows():
    xmin, xmax, ymin, ymax = box.xmin, box.xmax, box.ymin, box.ymax
    center_r, center_c = int((ymin + ymax)/2), int((xmin + xmax)/2)
    r = math.ceil((ymax-ymin + xmax-xmin)/4)
    val = 255
    #val = 50+205*(box.patch_size - 50)/1150 # for debug
    cv2.circle(out_img, [center_c,center_r], r, (val), -1)
  # morphology
  out_img = scipy.ndimage.morphology.binary_dilation(out_img,
                    structure=skimage.morphology.disk(model_inference_config['morphology_factor']))
  out_img = scipy.ndimage.morphology.binary_closing(out_img,
                    structure=skimage.morphology.disk(model_inference_config['morphology_factor']))
  #out_img = scipy.ndimage.morphology.binary_erosion(out_img,
  #                  structure=skimage.morphology.disk(1))
  out_img = np.where(out_img > 0, 255, 0)
  # save output
  with rasterio.open(out_img_path, 'w', driver='GTiff',
                  width=image_shape[1], height=image_shape[0],
                  count=1, dtype=rasterio.uint8, nodata=0,
                  transform=image_transform, crs=image_crs) as dst:
    dst.write(out_img.astype(np.uint8), 1)
  logging.debug("PROCESSED: {p}".format(p=out_img_path))

def postprocess_render_images(model_inference_config,
                              dataset_img_dir, dataset_inference_dir,
                              dataset_img_pattern="*.tiff",
                              continue_mode=True, **ignored):
  """
  Render tree canopy as raster images from detected bboxes.

  Parameters
  ----------
  model_inference_config : dict
    inference parameter
  dataset_img_dir : str
    the path to the source image dataset folder
  dataset_inference_dir : str
    the path to the inference folder
    The bboxes in *.pkl stored in the subfolder `b/` of the output folder as input
    will be rendered and the resulting raster images will be stored in the subfolder `r`
  dataset_img_pattern : str
    the pattern of filter to select images
  continue_mode : bool
    continue without working on already processed images
  """
  CONTINUE_MODE = continue_mode

  SRC_IMG_DIR           = Path(dataset_img_dir)
  SRC_BBOX_DIR          = Path(dataset_inference_dir).joinpath('b')
  OUTPUT_IMG_RESULT_DIR = Path(dataset_inference_dir).joinpath('r')
  MODEL_INFERENCE       = model_inference_config

  CONCURRENCY = MODEL_INFERENCE['concurrency']

  os.makedirs(OUTPUT_IMG_RESULT_DIR, exist_ok=True)

  print("source images:", SRC_IMG_DIR)
  print("source inferred bbox:", SRC_BBOX_DIR)
  print("output rendered raster:", OUTPUT_IMG_RESULT_DIR)

  tasks = []
  for src_img_path in SRC_IMG_DIR.glob(dataset_img_pattern):
    delayed = dask.delayed(postprocess_render_image)(src_img_path=src_img_path, src_bbox_dir=SRC_BBOX_DIR,
                                         output_img_dir=OUTPUT_IMG_RESULT_DIR,
                                         model_inference_config=MODEL_INFERENCE,
                                         continue_mode=CONTINUE_MODE)
    tasks.append(delayed)

  with dask.config.set(pool=ThreadPoolExecutor(CONCURRENCY)):
    dask.compute(*tasks)
