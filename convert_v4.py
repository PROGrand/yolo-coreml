import argparse
import os
#import cv2
from coremltools import ImageType
from yolov4.tf import YOLOv4
import coremltools as ct 
#import tensorflow as tf
#from coremltools.proto import NeuralNetwork_pb2

parser = argparse.ArgumentParser(description='Yolo4 To CoreML Converter.')
parser.add_argument('-n', '--names_path', help='Path to names file.')
parser.add_argument('-c', '--config_path', help='Path to Darknet cfg file.')
parser.add_argument('-w', '--weights_path', help='Path to Darknet weights file.')
parser.add_argument('-m', '--mlpackage_path', help='Path to output CoreML mlpackage file.')

yolo = YOLOv4()

def _main(args):
    names_path = os.path.expanduser(args.names_path)
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    mlpackage_path = os.path.expanduser(args.mlpackage_path)

    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)

    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    assert mlpackage_path.endswith(
        '.mlpackage'), 'output path {} is not a .mlpackage file'.format(mlpackage_path)

    print('names: ', names_path)
    print('config: ', config_path)
    print('weights: ', weights_path)
    print('mlpackage: ', names_path)

    yolo.config.parse_names(names_path)
    yolo.config.parse_cfg(config_path)

    yolo.make_model()
    yolo.load_weights(weights_path, weights_type="yolo")
    yolo.summary(summary_type="yolo")
    yolo.summary()


    # Convert to Core ML
    model = ct.convert(yolo.model,
                       inputs=[ImageType(name='input_1', scale=1/255., color_layout="BGR",
                                         channel_first=False)],
                       minimum_deployment_target=ct.target.iOS15,
                       compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.ALL,
                       skip_model_load=False,
                       debug=False
                       )

    model.save(mlpackage_path)

    #yolo.inference(media_path="kite.jpg")


if __name__ == '__main__':
    _main(parser.parse_args())