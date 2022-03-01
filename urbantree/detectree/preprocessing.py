import os
import pandas as pd
from pathlib import Path
import shutil
import detectree


def dataset_split_feature_cluster(dataset_train_dir,
                                  dataset_split_meta_path,
                                  dataset_split_nr_clusters,
                                  dataset_split_nr_pca_comp,
                                  dataset_split_train_proportion,
                                  dataset_img_meta_path=None,
                                  dataset_img_path_list=None,
                                  dataset_split_max_nodata_threshold=1.0,
                                  **ignored):
  """
  Split and extract training dataset according to pixel features such as color, texture and entropy.
  Selected training dataset will be copied to `dataset_train_dir` and
  the summary of the result is stored as CSV file with columns: ['img_filepath', 'img_cluster', 'train']

  Notes
  -----
  - if `dataset_img_path_list` is given, data exceeding `dataset_split_max_nodata_threshold` will be never choosen as traiing set.
  - Either `dataset_img_meta_path` or `dataset_img_path_list` must be defined.

  Parameters
  ----------
  dataset_train_dir : str
    output folder where selected training dataset will be copied to.
  dataset_split_meta_path : str
    output meta file path
  dataset_split_nr_clusters : int
    the number of clusters
  dataset_split_nr_pca_comp : int
    the number of PCA compoenents
  dataset_split_train_proportion : float
    proportion of train amount (0.0 - 1.0)
  dataset_img_meta_path : str
    the path to the meta file describing images as dataset
  dataset_img_path_list : list
    a list of image paths as dataset
  dataset_split_max_nodata_threshold : float
    data with nodata ratio higer than this threshold will never be choosen as training dataset
  """

  ################### Constants ###########################

  assert dataset_img_meta_path or dataset_img_path_list, "Either `dataset_img_meta_path` or `dataset_img_path_list` must be defined."

  DATASET_IMG_META_PATH  = Path(dataset_img_meta_path)
  MAX_NODATA_THRESHOLD   = dataset_split_max_nodata_threshold
  DATASET_IMG_PATH_LIST  = dataset_img_path_list

  # use default "cluster-II" method
  # we can still decide to use Cluter-I or Cluster-II methods during training
  NR_SPLIT_CLUSTERS = dataset_split_nr_clusters
  NR_PCA_COMPONENTS = dataset_split_nr_pca_comp
  TRAIN_PROPOTION   = dataset_split_train_proportion

  # training data and splitting result
  TRAIN_SPLIT_META_PATH = Path(dataset_split_meta_path)
  DATASET_TRAIN_DIR     = Path(dataset_train_dir)

  ##########################################################

  # reset folders
  os.makedirs(Path(TRAIN_SPLIT_META_PATH).parent, exist_ok=True)
  os.makedirs(DATASET_TRAIN_DIR, exist_ok=True)

  ## prepare source images to process
  if DATASET_IMG_META_PATH:
    meta_df = pd.read_csv(DATASET_IMG_META_PATH)
    threshold_filter = meta_df['nodata'] <= MAX_NODATA_THRESHOLD
    selected_img_filepaths   = meta_df[threshold_filter]['img_filepath'].tolist()
    unselected_img_filepaths = meta_df[~threshold_filter]['img_filepath'].tolist()
  else:
    selected_img_filepaths = DATASET_IMG_PATH_LIST
    unselected_img_filepaths = []

  if len(selected_img_filepaths) == 0:
    raise RuntimeError("No file is available to be training data.")

  # at first, process only selected images to split with "cluster-II" method
  ts = detectree.TrainingSelector(img_filepaths=selected_img_filepaths)
  split_df, evr = ts.train_test_split(method="cluster-II",
                                      num_components=NR_PCA_COMPONENTS,
                                      num_img_clusters=NR_SPLIT_CLUSTERS,
                                      train_prop=TRAIN_PROPOTION,
                                      return_evr=True)

  # then, assign unselected images (depending on threshold) to a separate cluster as test only data
  for img_filepath in unselected_img_filepaths:
    split_df = split_df.append({'img_filepath': img_filepath, 'img_cluster': NR_SPLIT_CLUSTERS, 'train': False}, ignore_index=True)

  # save split result
  split_df.to_csv(TRAIN_SPLIT_META_PATH, index=False)
  print("split summary: " + str(TRAIN_SPLIT_META_PATH))

  # make a copy of training data
  for index, d in split_df[split_df['train']].iterrows():
    shutil.copy(d.img_filepath, DATASET_TRAIN_DIR)
  print("Training dataset is copied to " + str(DATASET_TRAIN_DIR))