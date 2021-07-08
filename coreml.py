#! /usr/bin/env python

import sys
import coremltools
from coremltools.proto import NeuralNetwork_pb2

def convert_mish(layer):
	params = NeuralNetwork_pb2.CustomLayerParams()
	params.className = "Mish"
	params.description = "Mish Activation Layer"
	return params


if len(sys.argv) == 1:
	print("Please, specify model name.")
	sys.exit(-1)

coreml_model = coremltools.converters.keras.convert(
	sys.argv[1] + '.h5',
	input_names='image',
	image_input_names='image',
	image_scale=1/255.,
	add_custom_layers=True,
	custom_conversion_functions={ "Mish": convert_mish }
)

coreml_model.author = 'mtbo.org'
coreml_model.license = 'mtbo.org'
coreml_model.short_description = "The YOLOv4 network"

print(coreml_model)

coreml_model.save(sys.argv[1] + '.mlmodel')
