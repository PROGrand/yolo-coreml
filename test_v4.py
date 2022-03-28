import argparse
import os
import cv2
from yolov4.tf import YOLOv4
from os import path
import time
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Yolo4 Test.')
parser.add_argument('-n', '--names_path', help='Path to names file.')
parser.add_argument('-c', '--config_path', help='Path to Darknet cfg file.')
parser.add_argument('-w', '--weights_path', help='Path to Darknet weights file.')
parser.add_argument('-g', '--gray', help='Gray image.')
parser.add_argument('input_path', help='Path to input image.', nargs='+')

yolo = YOLOv4()


def predict(frame: np.ndarray, prob_thresh: float, gray: bool):
    height, width, _ = frame.shape

    image_data = yolo.resize_image(frame)
    
    if gray:
        color_convert = cv2.COLOR_BGR2GRAY
    else:
        color_convert = cv2.COLOR_BGR2RGB

    image_data = cv2.cvtColor(image_data, color_convert)

    if gray:
        height, width = image_data.shape
        image_data = image_data.reshape([height, width, 1])

    
    image_data = image_data / 255.0

    if gray:
        image_data = image_data.reshape(yolo.config.net.input_shape)

    image_data = image_data[np.newaxis, ...].astype(np.float32)

    candidates = yolo._predict(image_data)
    candidates = [
        c.numpy().astype(np.float32, copy=False) for c in candidates
    ]

    pred_bboxes = yolo.get_yolo_detections(
        yolos=candidates, prob_thresh=prob_thresh
    )
    yolo.fit_to_original(pred_bboxes, height, width)
    return pred_bboxes

def inference(
        media_path,
        prob_thresh: float = 0.12,
        gray: bool = False
):
    if isinstance(media_path, str) and not path.exists(media_path):
        raise FileNotFoundError("{} does not exist".format(media_path))

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

    frame = cv2.imread(media_path)

    start_time = time.time()
    bboxes = predict(frame, prob_thresh=prob_thresh, gray=gray)
    exec_time = time.time() - start_time
    print("time: {:.2f} ms".format(exec_time * 1000))

    image = yolo.draw_bboxes(frame, bboxes)
    cv2.imshow("result", image)
    print("YOLOv4: Inference is finished")
    cv2.waitKey(0)
    cv2.destroyWindow("result")

def _main(args):
    names_path = os.path.expanduser(args.names_path)
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    gray = args.gray

    tf.config.experimental.enable_mlir_graph_optimization()

    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)

    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    print('names: ', names_path)
    print('config: ', config_path)
    print('weights: ', weights_path)

    yolo.config.parse_names(names_path)
    yolo.config.parse_cfg(config_path)

    yolo.make_model()
    yolo.load_weights(weights_path, weights_type="yolo")
    yolo.summary()

    for file_name in args.input_path:
        input_path = os.path.expanduser(file_name)
        inference(media_path=input_path, gray=gray)


if __name__ == '__main__':
    _main(parser.parse_args())