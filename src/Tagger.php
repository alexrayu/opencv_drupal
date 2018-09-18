<?php

namespace Drupal\opencv;

use CV\Scalar;
use CV\Size;
use CV\Point;
use function CV\{imread, rectangle, putText, imwrite};
use function CV\DNN\{blobFromImage, readNetFromTensorflow};

/**
 * Class Tagger.
 */
class Tagger {

  /**
   * Constructs a new Tagger object.
   */
  public function __construct() {

  }

  /**
   * Tag the images.
   *
   * @param string $image_path
   *   Image path.
   * @param bool $draw
   *   Whether to draw border around found items.
   *
   * @return array
   *   Detected tags.
   */
  public function tag($image_path, $draw = FALSE) {
    $tags = [];
    $module_path = drupal_get_path('module', 'opencv');
    $categories = explode("\n", file_get_contents($module_path . '/models/ssdlite_mobilenet_v2_coco/classes.txt'));
    $src = imread($image_path);
    $blob = blobFromImage($src, 0.017, new Size(300, 300), new Scalar(127.5, 127.5, 127.5), TRUE, FALSE);
    $net = readNetFromTensorflow($module_path . '/models/ssdlite_mobilenet_v2_coco/frozen_inference_graph.pb', $module_path . '/models/ssdlite_mobilenet_v2_coco/ssdlite_mobilenet_v2_coco.pbtxt');
    $net->setInput($blob, "");
    $r = $net->forward();
    for ($i = 0; $i < $r->shape[2]; $i++) {
      $classId    = $r->atIdx([0, 0, $i, 1]);
      $confidence = intval($r->atIdx([0, 0, $i, 2]) * 100);
      if ($classId && $confidence > 50) {
        $tags[$categories[$classId]] = $categories[$classId];
        $startX = $r->atIdx([0, 0, $i, 3]) * $src->cols;
        $startY = $r->atIdx([0, 0, $i, 4]) * $src->rows;
        $endX = $r->atIdx([0, 0, $i, 5]) * $src->cols;
        $endY = $r->atIdx([0, 0, $i, 6]) * $src->rows;
        $scalar = new Scalar(0, 0, 255);
        rectangle($src, $startX, $startY, $endX, $endY, $scalar, 1);
        $text = "{$categories[$classId]} $confidence%";
        rectangle($src, $startX, $startY, $startX + 11 * strlen($text), $startY - 20, new Scalar(255, 255, 255), -2);
        putText($src, $text, new Point($startX + 5, $startY - 5), 1, 1, new Scalar(), 1);
      }
    }

    if ($draw) {
      imwrite($image_path, $src);
    }

    return $tags;
  }

}
