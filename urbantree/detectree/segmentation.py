import os
import pandas as pd
import logging
import detectree
import numpy
import rasterio
import inspect
from pathlib import Path
import maxflow as mf
import dask
from dask import diagnostics
from detectree import pixel_features, pixel_response
import joblib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn import ensemble

def generate_response(dataset_train_dir, dataset_response_dir,
                      **ignored):

  # the labelme result (*.json) which has to be processed with `labelme_json_to_dataset`
  # `labelme_json_to_dataset` generates a folder for each labeling result
  LABELME_RESULTS = list(Path(dataset_train_dir).glob('*.json'))

  # output folder
  RESPONSE_DETECTREE_IMG_PATH = dataset_response_dir

  os.makedirs(RESPONSE_DETECTREE_IMG_PATH, exist_ok=True)

  for lbme in LABELME_RESULTS:
    d = Path(lbme)
    # original feaure images
    feature_dir = d.parent
    feature_path = feature_dir.joinpath( d.stem + ".tiff" )
    # labelme result
    resp_src_dir = d.parent.joinpath( d.stem.replace('.', '_') + "_json" )
    resp_src = resp_src_dir.joinpath("label.png")
    # converted output
    resp_output = Path(RESPONSE_DETECTREE_IMG_PATH).joinpath( d.stem + ".tiff" )

    with rasterio.open(feature_path) as feature:
      with rasterio.open(resp_src) as response:
        data = response.read()
        data[data > 0] = 255
        with rasterio.open(resp_output, 'w', driver='GTiff',
                    width=data.shape[2], height=data.shape[1],
                    count=response.count, dtype=data.dtype, nodata=0,
                    transform=feature.transform, crs=feature.crs) as dst:
          dst.write(data)


def train_model(model_params_config,
                dataset_split_meta_path,
                dataset_response_dir,
                model_path, model_cluster_path_pattern,
                pretrained_model_path=None,
                **ignored):

  PARAMS_CLASSIFIER_TRAINER = model_params_config

  # train/test dataset
  TRAIN_SPLIT_RESULT_CSV = dataset_split_meta_path

  # response labeling
  RESPONSE_DETECTREE_IMG_PATH = dataset_response_dir

  # persistent trained classifier
  CLASSIFIER_C1_DUMP = model_path
  CLASSIFIER_C2_DUMP = model_cluster_path_pattern

  PRETRAINED_CLASSIFIER_PATH = pretrained_model_path

  os.makedirs(Path(CLASSIFIER_C1_DUMP).parent, exist_ok=True)
  os.makedirs(Path(CLASSIFIER_C2_DUMP).parent, exist_ok=True)

  split_df = pd.read_csv(TRAIN_SPLIT_RESULT_CSV)
  training_df = split_df[split_df["train"]]

  def train_classifier(trainer: detectree.ClassifierTrainer, split_df=None, response_img_dir=None,
                      method=None, img_cluster=None, pre_trained_clf : ensemble.AdaBoostClassifier=None):
      X = pixel_features.PixelFeaturesBuilder(
          **trainer.pixel_features_builder_kws).build_features(
              split_df=split_df, method=method, img_cluster=img_cluster)
      y = pixel_response.PixelResponseBuilder(
          **trainer.pixel_response_builder_kws).build_response(
              split_df=split_df, response_img_dir=response_img_dir,
              method=method, img_cluster=img_cluster)
      clf = pre_trained_clf if pre_trained_clf is not None else \
              ensemble.AdaBoostClassifier(n_estimators=trainer.num_estimators,
                                          **trainer.adaboost_kws)
      clf.fit(X, y)
      return clf

  def train_classifiers(trainer: detectree.ClassifierTrainer, split_df, response_img_dir, pre_trained_clf : ensemble.AdaBoostClassifier=None):
      assert 'img_cluster' in split_df
      clfs_lazy = {}
      for img_cluster, _ in split_df.groupby('img_cluster'):
          clfs_lazy[img_cluster] = dask.delayed(train_classifier)(
              trainer=trainer,
              split_df=split_df, response_img_dir=response_img_dir,
              method='cluster-II', img_cluster=img_cluster,
              pre_trained_clf=pre_trained_clf)
      with diagnostics.ProgressBar():
          clfs_dict = dask.compute(clfs_lazy)[0]
      return clfs_dict

  pre_trained_clf = None
  if PRETRAINED_CLASSIFIER_PATH is not None:
      pre_trained_clf = joblib.load(PRETRAINED_CLASSIFIER_PATH)

  # train classifier (cluster-II method)
  clfs_c2 = train_classifiers(detectree.ClassifierTrainer(**PARAMS_CLASSIFIER_TRAINER),
                              split_df=training_df,
                              response_img_dir=RESPONSE_DETECTREE_IMG_PATH,
                              pre_trained_clf=pre_trained_clf)
  # save trained classifier
  for c, clf in clfs_c2.items():
    joblib.dump(clf, CLASSIFIER_C2_DUMP.format(i=c))

  # train classifier (cluster-I method)
  clf_c1 = train_classifier(detectree.ClassifierTrainer(**PARAMS_CLASSIFIER_TRAINER),
                              split_df=training_df,
                              response_img_dir=RESPONSE_DETECTREE_IMG_PATH,
                              method='cluster-I',
                              pre_trained_clf=pre_trained_clf)
  # save trained classifier
  joblib.dump(clf_c1, CLASSIFIER_C1_DUMP)


def save_image(data, path, img_transform, img_crs, dtype, astype):
  with rasterio.open(path, 'w', driver='GTiff',
                  width=data.shape[1], height=data.shape[0],
                  count=1, dtype=dtype, nodata=0,
                  transform=img_transform, crs=img_crs) as dst:
    dst.write(data.astype(astype), 1)


def infer_image(clf, src_path, result_proba_path,
                model_params_config,
                continue_mode=True):

  CONTINUE_MODE = continue_mode
  MODEL_PARAMS = model_params_config
  MODEL_PARAMS_FEATURES = {k: v for k, v in MODEL_PARAMS.items() if k in inspect.getfullargspec(pixel_features.PixelFeaturesBuilder).args}

  if result_proba_path.exists() and CONTINUE_MODE:
    logging.debug("Passed: {f}".format(f=src_path.stem))
    return

  # meta
  src = rasterio.open(src_path)
  img_shape = src.shape
  img_transform = src.transform
  img_crs = src.crs
  src.close()

  # pixel feature
  X = pixel_features.PixelFeaturesBuilder(**MODEL_PARAMS_FEATURES) \
        .build_features_from_filepath(src_path)
  # predict raw probability of all pixel w.r.t tree and none-tree
  p_nontree, p_tree = numpy.hsplit(clf.predict_proba(X), 2)

  P_nontree = p_nontree.reshape(img_shape)
  P_tree = p_tree.reshape(img_shape)
  save_image(data=P_tree, path=result_proba_path, img_transform=img_transform, img_crs=img_crs, dtype=rasterio.float32, astype=numpy.float32)
  logging.debug("Processed: {f}".format(f=src_path.stem))


def infer_images(dataset_img_dir, dataset_inference_dir,
                 dataset_split_meta_path, model_inference_config,
                 model_path, model_cluster_path_pattern,
                 model_params_config,
                 continue_mode=True,
                 **ignored):
  CONTINUE_MODE = continue_mode

  DATASET_IMG_DIR = Path(dataset_img_dir).parent
  PREDICT_RESULT_DIR = Path(dataset_inference_dir)
  PREDICT_PROBA_RESULT_DIR = PREDICT_RESULT_DIR.joinpath('p')
  TRAIN_SPLIT_RESULT_CSV = Path(dataset_split_meta_path)

  # pre-trained model
  CLASSIFIER_DETECTREE_C1_DUMP  = model_path
  CLASSIFIER_DETECTREE_C2_DUMP  = model_cluster_path_pattern
  # model parameter
  MODEL_PARAMS = model_params_config
  CONCURRENCY = model_inference_config['concurrency']

  os.makedirs(PREDICT_PROBA_RESULT_DIR, exist_ok=True)

  # load all dataset
  split_df = pd.read_csv(TRAIN_SPLIT_RESULT_CSV)

  # load all classifier
  classifier = {}
  for c in split_df['img_cluster'].unique():
    dump = CLASSIFIER_DETECTREE_C2_DUMP.format(i=c)
    if Path(dump).exists():
      classifier[c] = joblib.load(dump)
    else:
      print("The cluster '{c}' has no available trained model, a generic one will be used instead.".format(c=c))
      classifier[c] = joblib.load(CLASSIFIER_DETECTREE_C1_DUMP)

  tasks = []
  for index, d in split_df.iterrows():
    src_path = Path(d.img_filepath)
    result_proba_path = PREDICT_PROBA_RESULT_DIR.joinpath(src_path.name)
    clf = classifier[d.img_cluster]
    delayed = dask.delayed(infer_image)(clf=clf,
                src_path=src_path, result_proba_path=result_proba_path,
                model_params_config=MODEL_PARAMS,
                continue_mode=CONTINUE_MODE)

    tasks.append(delayed)

  with dask.config.set(pool=ThreadPoolExecutor(CONCURRENCY)):
    dask.compute(*tasks)

def postprocess_render_image(src_proba_path, output_render_dir,
                             model_params_config, model_inference_config,
                             continue_mode=True):

  CONTINUE_MODE = continue_mode
  PREDICT_RENDER_RESULT_DIR = output_render_dir
  MODEL_PARAMS = model_params_config
  MODEL_INFERENCE_PARAMS = model_inference_config

  result_img_path = PREDICT_RENDER_RESULT_DIR.joinpath(src_proba_path.name)

  if result_img_path.exists() and CONTINUE_MODE:
    logging.debug("Passed: {f}".format(src_proba_path.stem))
    return

  # pixel probability image
  src = rasterio.open(src_proba_path)
  img_shape = src.shape # (H, W)
  img_transform = src.transform
  img_crs = src.crs
  P_tree = src.read()
  P_tree = P_tree[0,:,:] # convert (C, H, W) to (H, W)
  P_nontree = 1.0 - P_tree
  src.close()

  # refinement with min-cut/max-flow based on raw pixel probability
  g = mf.Graph[int]()
  node_ids = g.add_grid_nodes(img_shape)
  D_tree = (MODEL_INFERENCE_PARAMS['refine_int_rescale'] * numpy.log(P_nontree)).astype(int)
  D_nontree = (MODEL_INFERENCE_PARAMS['refine_int_rescale'] * numpy.log(P_tree)).astype(int)
  MOORE_NEIGHBORHOOD_ARR = numpy.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
  g.add_grid_edges(node_ids, MODEL_INFERENCE_PARAMS['refine_beta'],
                    structure=MOORE_NEIGHBORHOOD_ARR)
  g.add_grid_tedges(node_ids, D_tree, D_nontree)
  g.maxflow()
  refined = numpy.full(img_shape, MODEL_PARAMS['nontree_val'])
  refined[g.get_grid_segments(node_ids)] = MODEL_PARAMS['tree_val']
  save_image(refined, path=result_img_path, img_transform=img_transform, img_crs=img_crs, dtype=rasterio.uint8, astype=numpy.uint8)

  logging.debug("Processed: {f}".format(f=src_proba_path.stem))

def postprocess_render_images(dataset_inference_dir,
                              model_params_config, model_inference_config,
                              dataset_img_pattern="*.tiff",
                              continue_mode=True, **ignored):
  CONTINUE_MODE = continue_mode

  PREDICT_RESULT_DIR = Path(dataset_inference_dir)
  PREDICT_RENDER_RESULT_DIR = PREDICT_RESULT_DIR.joinpath('r')
  PREDICT_PROBA_SRC_DIR     = PREDICT_RESULT_DIR.joinpath('p')

  MODEL_PARAMS           = model_params_config
  MODEL_INFERENCE_PARAMS = model_inference_config

  CONCURRENCY = MODEL_INFERENCE_PARAMS['concurrency']

  os.makedirs(PREDICT_RENDER_RESULT_DIR, exist_ok=True)

  tasks = []

  for src_proba_path in PREDICT_PROBA_SRC_DIR.glob(dataset_img_pattern):
    delayed = dask.delayed(postprocess_render_image)(src_proba_path=src_proba_path,
                             output_render_dir=PREDICT_RENDER_RESULT_DIR,
                             model_params_config=MODEL_PARAMS,
                             model_inference_config=MODEL_INFERENCE_PARAMS,
                             continue_mode=CONTINUE_MODE)
    tasks.append(delayed)

  with dask.config.set(pool=ThreadPoolExecutor(CONCURRENCY)):
    dask.compute(*tasks)