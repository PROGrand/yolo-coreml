#!/bin/sh

sed 's/subdivisions.*//
  s/decay.*//
  s/angle.*//
  s/saturation.*//
  s/exposure.*//
  s/hue.*//
  s/flip.*//
  s/small_object.*//
  s/learning_rate.*//
  s/burn_in.*//
  s/max_batches.*//
  s/policy.*//
  s/steps.*//
  s/scales.*//

  s/jitter.*//
  s/ignore_thresh.*//
  s/truth_thresh.*//
  s/random.*//
  s/resize.*//
  s/max_delta.*//
  
  s/stopbackward.*//
  ' $1 > $2