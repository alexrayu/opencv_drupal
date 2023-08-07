<?php

namespace Drupal\opencv;

use CV\Scalar;
use function CV\{imread, imwrite, circle, cvtColor, equalizeHist};
use CV\CascadeClassifier, CV\Face\FacemarkLBF;
use const CV\{COLOR_BGR2GRAY};
use Drupal\Core\Extension\ExtensionPathResolver;

/**
 * Class FaceFinder.
 */
class FaceFinderLBP {

  /**
   * The extension path resolver.
   *
   * @var \Drupal\Core\Extension\ExtensionPathResolver
   */
  protected $extensionPathResolver;

  /**
   * Sets the extension path resolver.
   *
   * @param \Drupal\Core\Extension\ExtensionPathResolver $extension_path_resolver
   *   The extension path resolver.
   */
  public function setExtensionPathResolver(ExtensionPathResolver $extension_path_resolver) {
    $this->extensionPathResolver = $extension_path_resolver;
  }

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
    $module_path = $this->setExtensionPathResolver;

    $src = imread($image_path);

    $gray = cvtColor($src, COLOR_BGR2GRAY);
    equalizeHist($gray, $gray);

    $faceClassifier = new CascadeClassifier();
    $faceClassifier->load($module_path . '/models/lbpcascades/lbpcascade_frontalface.xml');
    $faces = null;
    $faceClassifier->detectMultiScale($gray, $faces);
    $facemark = FacemarkLBF::create();
    $facemark->loadModel($module_path . '/models/opencv-facemark-lbf/lbfmodel.yaml');

    $facemark->fit($src, $faces, $landmarks);
    if ($landmarks) {
      $scalar = new Scalar(0, 0, 255);
      foreach ($landmarks as $face) {
        foreach($face as $k => $point) {
          circle($src, $point, 1, $scalar, 2);
        }
      }
    }

    imwrite($image_path, $src);
  }

}
