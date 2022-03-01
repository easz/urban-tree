
## Folder structure

    |- aerial_images/
    |  |- {SOURCE_MAP_TILE_NAME}/      # original web map tiles in XYZ structure as image source
    |                                  # (e.g. retrieved with `wms-tile-get`)
    |
    |- aerial_images_resampled/
    |  |- {DATASET_NAME}/              # resampled map tiles in arbitrary sizes as dataset
    |                                  # (e.g. with wf-resample-map-tiles.ipynb)
    |
    |- setting/                        # setting files
    |  |- {DATASET_NAME}/
    |     |- {PROFILE}/
    |
    |- interim/
       |- {DATASET_NAME}/
          |- {PROFILE}/
             |- RADME.md               # description of the profile
             |- meta/                  # meta info (e.g. resampled dataset, split result)
             |- model/                 # trained models (e.g. classifier, network)
             |- inference/             # inference result
             |- response/              # response data for supervised training
             |- train/                 # training data with annotation
