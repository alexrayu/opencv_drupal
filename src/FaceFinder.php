<?php

namespace Drupal\opencv;

use CV\Scalar;
use CV\Size;
use function CV\{imread, rectangle, imwrite};
use function CV\DNN\{blobFromImage, readNetFromCaffe};

/**
 * Class FaceFinder.
 */
class FaceFinder {

  /**
   * Constructs a new FaceFinder object.
   */
  public function __construct() {

  }

  /**
   * Find faces in the image.
   *
   * @param string $image_path
   *   Image path.
   */
  public function find($image_path) {
    $module_path = drupal_get_path('module', 'opencv');
    $src = imread($image_path);
    $blob = blobFromImage($src, 1, new Size(), new Scalar(104, 177, 123),
      TRUE, FALSE);
    $net = readNetFromCaffe($module_path . '/models/ssd/res10_300x300_ssd_deploy.prototxt',
      $module_path . '/models/ssd/res10_300x300_ssd_iter_140000.caffemodel');
    $net->setInput($blob, "");
    $r = $net->forward();
    $scalar = new Scalar(0, 255, 100);
    for ($i = 0; $i < $r->shape[2]; $i++) {
      $confidence = $r->atIdx([0, 0, $i, 2]);
      if ($confidence > 0.5) {
        $startX = $r->atIdx([0, 0, $i, 3]) * $src->cols;
        $startY = $r->atIdx([0, 0, $i, 4]) * $src->rows;
        $endX   = $r->atIdx([0, 0, $i, 5]) * $src->cols;
        $endY   = $r->atIdx([0, 0, $i, 6]) * $src->rows;

        rectangle($src, $startX, $startY, $endX, $endY, $scalar, 1);
      }
    }

    imwrite($image_path, $src);
  }

}
