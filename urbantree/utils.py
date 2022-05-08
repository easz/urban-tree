import supermercado
import mercantile


def geerate_tile_def_from_feature(features, zooms, projected='mercator'):
  """
      yield [x, y, z, xmin, ymin, xmax, ymax]
      @param features
              a list  geojson features (i.e. polygon) objects
      @param zooms
              a list of zoom levels
      @param projected
              'mercator' or 'geographic'
  """
  # $ supermercado burn <zoom>
  features = [f for f in supermercado.super_utils.filter_features(features)]
  for zoom in zooms:
    zr = zoom.split("-")
    for z in range(int(zr[0]),  int(zr[1] if len(zr) > 1 else zr[0]) + 1):
      for t in supermercado.burntiles.burn(features, z):
        tile = t.tolist()
        # $ mercantile shapes --mercator
        feature = mercantile.feature(
            tile,
            fid=None,
            props={},
            projected=projected,
            buffer=None,
            precision=None
        )
        bbox = feature["bbox"]
        yield tile + bbox