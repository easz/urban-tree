import math
import mercantile
import numpy
import os
import pandas as pd
from pathlib import Path
import rasterio
from tqdm import tqdm
import glob

def dataset_resample_xyz_tiles(source_map_tile_xyz_pattern,
                               source_map_tile_zoom,
                               dataset_area_bbox,
                               dataset_img_dir,
                               dataset_img_meta_path=None,
                               dataset_img_width=512,
                               dataset_img_height=512,
                               dataset_img_overlapping=64,
                               **ignored):
  """
  Resample images from XYZ map tiles

  - Reconstruct map tile images with specific size and overlapping.
  - usual XYZ map tile dataset as source (EPSG:3857)
  - GeoTiff output
  - A suumary of results is stored as CSV file with columns: ['img_filepath', 'nodata']

  Parameters
  ----------
  source_map_tile_xyz_pattern : str
    map tile image path pattern (e.g. `/some/dir/{z}/{x}/{y}.png`)
  source_map_tile_zoom : int
    specified zoom level to use
  dataset_area_bbox : list
    bbox of the area in epsg:3857
  dataset_img_dir : str
    output folder of re-sampled image
  dataset_img_meta_path : str
    output dataset meta file path
  dataset_img_width : int
    output dataset image width in pixel
  dataset_img_height : int
    output dataset image height in pixel
  dataset_img_overlapping : int
    output dataset image overlapping extent in pixel
 """
  ################### Constants ###########################

  ## source web map tiles
  SOURCE_MAP_TILE_ZOOM          = source_map_tile_zoom
  SOURCE_MAP_TILE_PATH_PATTERN  = source_map_tile_xyz_pattern
  SOURCE_MAP_TILE_WIDTH_PX      = None
  SOURCE_MAP_TILE_HEIGHT_PX     = None
  SOURCE_MAP_TILE_BAND_COUNT    = None
  SOURCE_MAP_TILE_DTYPE         = None
  SOURCE_MAP_TILE_EPSG          = None

  ## sampling area bbox in epsg:3857 [left, bottom, right, top]
  DATASET_AREA_BBOX             = dataset_area_bbox

  ## output tile images
  DATASET_IMG_DIR               = Path(dataset_img_dir)
  DATASET_IMG_META_PATH         = Path(dataset_img_meta_path) or Path(dataset_img_dir).joinpath('dataset.csv')
  DATASET_IMG_WIDTH_PX          = dataset_img_width
  DATASET_IMG_HEIGHT_PX         = dataset_img_height
  DATASET_IMG_OVERLAPPING_PX    = dataset_img_overlapping

  SRC_MAP_TILE_GLOB_PATTERN = SOURCE_MAP_TILE_PATH_PATTERN.format(x="*", y="*", z="*")
  src_map_tile_list = glob.glob(SRC_MAP_TILE_GLOB_PATTERN)
  if len(src_map_tile_list) == 0:
    raise Exception("No file is found with the search pattern: {p}".format(p=SRC_MAP_TILE_GLOB_PATTERN))
  with rasterio.open(src_map_tile_list[0]) as peek:
    SOURCE_MAP_TILE_WIDTH_PX   = peek.meta['width']
    SOURCE_MAP_TILE_HEIGHT_PX  = peek.meta['height']
    SOURCE_MAP_TILE_BAND_COUNT = peek.meta['count']
    SOURCE_MAP_TILE_DTYPE      = peek.meta['dtype']
    SOURCE_MAP_TILE_EPSG       = peek.meta['crs'] or '3857'

  DATASET_IMG_PATH_PATTERN = DATASET_IMG_DIR / '{l:.9f}_{b:.9f}_{r:.9f}_{t:.9f}.tiff' # bbox as name

  DATASET_IMG_META = {
    'driver': 'GTiff',
    'dtype':  SOURCE_MAP_TILE_DTYPE,
    'width':  DATASET_IMG_WIDTH_PX,
    'height': DATASET_IMG_HEIGHT_PX,
    'count':  SOURCE_MAP_TILE_BAND_COUNT,
    'crs':    rasterio.crs.CRS.from_epsg(SOURCE_MAP_TILE_EPSG),
  }

  ###############################################################

  print("Processing XYZ map tiles from " + str(SRC_MAP_TILE_GLOB_PATTERN))

  # create output folder if necessary
  os.makedirs(DATASET_IMG_DIR, exist_ok=True)
  os.makedirs(Path(DATASET_IMG_META_PATH).parent, exist_ok=True)

  print("Resampled dataset will be saved in " + str(DATASET_IMG_DIR))

  # meta info to collect
  meta_img_filepath = []
  meta_nodata = []

  # process from upper-left to bottom-down
  UPPER_LEFT = mercantile.lnglat(DATASET_AREA_BBOX[0], DATASET_AREA_BBOX[3])
  UPPER_LEFT_TILE = mercantile.tile(UPPER_LEFT.lng, UPPER_LEFT.lat, SOURCE_MAP_TILE_ZOOM)

  # calculate all possible samples and create a progress bar
  BOTTOM_RIGHT = mercantile.lnglat(DATASET_AREA_BBOX[2], DATASET_AREA_BBOX[1])
  BOTTOM_RIGHT_TILE = mercantile.tile(BOTTOM_RIGHT.lng, BOTTOM_RIGHT.lat, SOURCE_MAP_TILE_ZOOM)
  TOTAL = math.ceil((SOURCE_MAP_TILE_WIDTH_PX * (BOTTOM_RIGHT_TILE.x - UPPER_LEFT_TILE.x + 1) - DATASET_IMG_OVERLAPPING_PX) / (DATASET_IMG_WIDTH_PX - DATASET_IMG_OVERLAPPING_PX)) \
        * math.ceil((SOURCE_MAP_TILE_HEIGHT_PX * (BOTTOM_RIGHT_TILE.y - UPPER_LEFT_TILE.y + 1) - DATASET_IMG_OVERLAPPING_PX) / (DATASET_IMG_HEIGHT_PX - DATASET_IMG_OVERLAPPING_PX))
  pbar = tqdm(total=TOTAL)

  tile_x = UPPER_LEFT_TILE.x
  x = DATASET_AREA_BBOX[0]
  bound = mercantile.xy_bounds(tile_x, UPPER_LEFT_TILE.y, SOURCE_MAP_TILE_ZOOM)
  px_x = int(math.floor((x - bound.left) * SOURCE_MAP_TILE_WIDTH_PX / (bound.right - bound.left)))
  while x < DATASET_AREA_BBOX[2]:

    tile_y = UPPER_LEFT_TILE.y
    y = DATASET_AREA_BBOX[3]
    bound = mercantile.xy_bounds(tile_x, tile_y, SOURCE_MAP_TILE_ZOOM)
    px_y = int(math.floor((y - bound.top) * SOURCE_MAP_TILE_HEIGHT_PX / (bound.bottom - bound.top)))

    nr_tile_x = int(math.ceil((px_x + DATASET_IMG_WIDTH_PX ) / SOURCE_MAP_TILE_WIDTH_PX))

    while y > DATASET_AREA_BBOX[1]:

      nr_tile_y = int(math.ceil((px_y + DATASET_IMG_HEIGHT_PX) / SOURCE_MAP_TILE_HEIGHT_PX))

      # init dest tile image
      temp = numpy.zeros((SOURCE_MAP_TILE_BAND_COUNT, SOURCE_MAP_TILE_HEIGHT_PX*nr_tile_y, SOURCE_MAP_TILE_WIDTH_PX*nr_tile_x), dtype=SOURCE_MAP_TILE_DTYPE)
      tile = numpy.zeros((SOURCE_MAP_TILE_BAND_COUNT, DATASET_IMG_HEIGHT_PX, DATASET_IMG_WIDTH_PX), dtype=SOURCE_MAP_TILE_DTYPE)

      # collect and merge sub src tiles
      has_tile_data = 0
      for i_x in range(nr_tile_x):
        for i_y in range(nr_tile_y):
          try:
            with rasterio.open(SOURCE_MAP_TILE_PATH_PATTERN.format(x=tile_x+i_x, y=tile_y+i_y, z=SOURCE_MAP_TILE_ZOOM)) as tile_src:
              data_src = tile_src.read()
              # src -> temp
              temp[:, i_y*SOURCE_MAP_TILE_HEIGHT_PX:(i_y+1)*SOURCE_MAP_TILE_HEIGHT_PX, i_x*SOURCE_MAP_TILE_WIDTH_PX :(i_x+1)*SOURCE_MAP_TILE_WIDTH_PX] = data_src[:,:,:]
              has_tile_data += 1
          except rasterio.errors.RasterioIOError as e:
            pass

      # progress with one sample
      pbar.update(1)

      # store the dest tile result
      if has_tile_data:
        tile[:,:,:] = temp[:, px_y:px_y+DATASET_IMG_HEIGHT_PX, px_x:px_x+DATASET_IMG_WIDTH_PX]
        tl = mercantile.xy_bounds(tile_x, tile_y, SOURCE_MAP_TILE_ZOOM)
        br = mercantile.xy_bounds(tile_x+nr_tile_x-1, tile_y+nr_tile_y-1, SOURCE_MAP_TILE_ZOOM)
        temp_tf = rasterio.transform.from_bounds(tl.left, br.bottom, br.right, tl.top, SOURCE_MAP_TILE_WIDTH_PX*nr_tile_x, SOURCE_MAP_TILE_HEIGHT_PX*nr_tile_y)
        win = rasterio.windows.Window(px_x, px_y, DATASET_IMG_WIDTH_PX, DATASET_IMG_HEIGHT_PX)
        tile_tf = rasterio.windows.transform(win, temp_tf)
        bl = tile_tf * [0, DATASET_IMG_HEIGHT_PX]
        tr = tile_tf * [DATASET_IMG_WIDTH_PX, 0]
        tile_out_path = str(DATASET_IMG_PATH_PATTERN).format(l=bl[0], b=bl[1],
                                                        r=tr[0], t=tr[1])
        with rasterio.open(tile_out_path, "w", **DATASET_IMG_META, transform=tile_tf) as dest:
          dest.write(tile)

        # collect meta info
        meta_img_filepath.append(tile_out_path)
        meta_nodata.append(1.0 - has_tile_data / (nr_tile_x*nr_tile_y))

      tile_y = tile_y + int(math.floor((px_y + DATASET_IMG_HEIGHT_PX - DATASET_IMG_OVERLAPPING_PX) / SOURCE_MAP_TILE_HEIGHT_PX))
      px_y = (px_y + DATASET_IMG_HEIGHT_PX - DATASET_IMG_OVERLAPPING_PX) % SOURCE_MAP_TILE_HEIGHT_PX
      bound = mercantile.xy_bounds(tile_x, tile_y, SOURCE_MAP_TILE_ZOOM)
      y = bound.top + px_y * (bound.bottom - bound.top) / SOURCE_MAP_TILE_HEIGHT_PX

    tile_x = tile_x + int(math.floor((px_x + DATASET_IMG_WIDTH_PX - DATASET_IMG_OVERLAPPING_PX) / SOURCE_MAP_TILE_WIDTH_PX))
    px_x = (px_x + DATASET_IMG_WIDTH_PX - DATASET_IMG_OVERLAPPING_PX) % SOURCE_MAP_TILE_WIDTH_PX
    bound = mercantile.xy_bounds(tile_x, tile_y, SOURCE_MAP_TILE_ZOOM)
    x = bound.left + px_x * (bound.right - bound.left) / SOURCE_MAP_TILE_WIDTH_PX

  # close the progress bar
  pbar.close()

  # create meta info (dataframe in csv)
  meta_df = pd.DataFrame({'img_filepath': meta_img_filepath, 'nodata': meta_nodata})
  meta_df.to_csv(DATASET_IMG_META_PATH, index=False)
  print("dataset summary: " + str(DATASET_IMG_META_PATH))
