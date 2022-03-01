import yaml
import numpy as np
from pydantic.utils import deep_update

class Setting(object):

  @staticmethod
  def load_deepforest_setting(file):
    """ load setting from a YAML file """
    src = yaml.safe_load(open(file).read())
    base = Setting.load_base_setting(file)
    default = Setting.__add_default_deepforest_setting(src=src, base=base)
    return deep_update(default, src)


  @staticmethod
  def load_detectree_setting(file):
    """ load setting from a YAML file """
    src = yaml.safe_load(open(file).read())
    base = Setting.load_base_setting(file)
    default = Setting.__add_default_detectree_setting(src=src, base=base)
    return deep_update(default, src)

  @staticmethod
  def __add_default_deepforest_setting(src, base):
    """ create default deepforest settings
    Parameters:
    -----------
    src : dict
      source setting definition from user
    base : dict
      required base setting

    Returns:
    --------
    dict
      constructed settings
    """
    DEFAULT_CROP_DIR = "crop" \
      if len(src.get('model_trainer_patch_sizes', [])) > 0 \
      else ""

    DEFAULT_TRAIN_ANNOTATION_FILE = "combined.all.csv_" \
      if src.get('model_trainer_train_on_validation_data', True) \
      else "combined.train.csv_"

    DEFAULT_PARAMS = {
      **base,
      'model_trainer_training_img_dir':
        '{dataset_response_dir}/{crop_dir}'.format(**base, crop_dir=DEFAULT_CROP_DIR),
      'model_trainer_training_annotation_path':
      '{dataset_response_dir}/{crop_dir}/{train_anno}'.format(**base, crop_dir=DEFAULT_CROP_DIR, train_anno=DEFAULT_TRAIN_ANNOTATION_FILE),
      'model_trainer_validation_annotation_path':
        '{dataset_response_dir}/{crop_dir}/combined.valid.csv_'.format(**base, crop_dir=DEFAULT_CROP_DIR)
    }

    DEFAULT_SETTING = {
      **base,

      'model_path': src.get('model_path', '{model_dir}/model.ckpt'.format(**DEFAULT_PARAMS)),

      'model_trainer_patch_sizes':              [],
      'model_trainer_patch_overlap_size':       32,
      'model_trainer_min_bbox_size':            0,
      'model_trainer_min_bbox_ratio':           0.3,
      'model_trainer_validation_ratio':         0.2,
      'model_trainer_train_on_validation_data': True,

      ## model
      'model_params_config': {
        'workers': 1,
        'gpus': 0,
        'distributed_backend': False,
        'batch_size': 1,
        'nms_thresh': 0.6,
        'score_thresh': 0.1,
        'train': {
          'fast_dev_run': False,
          'epochs': 20,
          'lr': 0.001,
          'preload_images': False,
          'root_dir': '{model_trainer_training_img_dir}'.format(**DEFAULT_PARAMS),
          'csv_file': '{model_trainer_training_annotation_path}'.format(**DEFAULT_PARAMS)
        },
        'validation': {
          'iou_threshold': 0.4,
          'val_accuracy_interval': 1,
          'root_dir': '{model_trainer_training_img_dir}'.format(**DEFAULT_PARAMS),
          'csv_file': '{model_trainer_validation_annotation_path}'.format(**DEFAULT_PARAMS)
        }
      },

      ## model inference
      'model_inference_config': {
        'confident_min_bbox_size': 0,
        'confident_min_score': 0.9,
        'morphology_factor': 0,
        'concurrency': 1,
        'patch': []
      }
    }
    return DEFAULT_SETTING

  @staticmethod
  def __add_default_detectree_setting(src, base):
    """ create default deepforest settings
    Parameters:
    -----------
    src : dict
      source setting definition from user
    base : dict
      required base setting

    Returns:
    --------
    dict
      constructed settings
    """
    DEFAULT_PARAMS = {
      **base
    }

    DEFAULT_SETTING = {
      **base,

      ## dataset splitting
      # train an image which has lower no-data ratio than the give threshold
      #(use '0' to train images without any nodata. use '1' to train all possible images)
      'dataset_split_max_nodata_threshold': 1.0,
      # the detectree's "cluster-II" method is used by default to split data set
      # However, we can still decide to use Cluster-I or Cluster-II methods during training and classifying
      'dataset_split_nr_clusters':      4,
      'dataset_split_nr_pca_comp':      12,
      'dataset_split_train_proportion': 0.01,
      'dataset_split_meta_path':        '{dataset_meta_dir}/split.csv'.format(**base),

      'model_path':                 '{model_dir}/model.joblib'.format(**base),
      'model_cluster_path_pattern': '{model_dir}/model_cluster_{i}.joblib'.format(i='{i}',**base),

      'model_params_config': {
        # feature
        'sigmas': [1, np.sqrt(2), 2],
        'min_neighborhood_range': 2,
        'num_neighborhoods': 3,
        # training
        'tree_val': 255,
        'nontree_val': 0,
        'num_estimators': 200,
        'learning_rate': 1.0,
      },
      'model_inference_config': {
        'refine': True,
        'refine_beta': 100,
        'refine_int_rescale': 10000,
        'concurrency' : 4
      }
    }
    return DEFAULT_SETTING

  @staticmethod
  def load_base_setting(file):
    """ construct base setting object from a dict object """
    src = yaml.safe_load(open(file).read())

    # mandatory IDs
    DEFAULT_IDS = {
      'dataset_name': src['dataset_name'],
      'profile_name': src['profile_name'],
    }
    # default folders
    DEFAULT_DIRS = {
      'dataset_img_dir':           src.get('dataset_img_dir',           'aerial_images_resampled/{dataset_name}/'         .format(**DEFAULT_IDS)),
      'dataset_meta_dir':          src.get('dataset_meta_dir',          'interim/{dataset_name}/{profile_name}/meta/'     .format(**DEFAULT_IDS)),
      'dataset_train_dir':         src.get('dataset_train_dir',         'interim/{dataset_name}/{profile_name}/train/'    .format(**DEFAULT_IDS)),
      'dataset_response_dir':      src.get('dataset_response_dir',      'interim/{dataset_name}/{profile_name}/response/' .format(**DEFAULT_IDS)),
      'dataset_inference_dir':     src.get('dataset_inference_dir',     'interim/{dataset_name}/{profile_name}/inference/'.format(**DEFAULT_IDS)),
      'model_dir':                 src.get('model_dir',                 'interim/{dataset_name}/{profile_name}/model/'    .format(**DEFAULT_IDS)),
      'model_trainer_logging_dir': src.get('model_trainer_logging_dir', 'interim/{dataset_name}/{profile_name}/log/'      .format(**DEFAULT_IDS)),
    }
    DEFAULT_PARAMS = { **DEFAULT_IDS, **DEFAULT_DIRS }
    DEFAULT_SETTING = {
      **DEFAULT_IDS,
      **DEFAULT_DIRS,

      ## source XYZ map tils
      'source_map_tile_zoom':        src.get('source_map_tile_zoom'),
      'source_map_tile_xyz_pattern': src.get('source_map_tile_xyz_pattern'),

      ## dataset image resmapling
      # resampled dataset area bbox in epsg:3857 [left, bottom, right, top]
      'dataset_area_bbox':       src.get('dataset_area_bbox'),
      'dataset_img_meta_path':   src.get('dataset_img_meta_path',      '{dataset_meta_dir}/dataset.csv'.format(**DEFAULT_PARAMS)),
      'dataset_img_width':       src.get('dataset_img_width',       512),
      'dataset_img_height':      src.get('dataset_img_height',      512),
      'dataset_img_overlapping': src.get('dataset_img_overlapping', 32),

      ## model
      'pretrained_model_path': src.get('pretrained_model_path')
    }

    return DEFAULT_SETTING