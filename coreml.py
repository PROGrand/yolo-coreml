#! /usr/bin/env python

import sys
import coremltools


if len(sys.argv) == 1:
	print("Please, specify model name.")
	sys.exit(-1)

coreml_model = coremltools.converters.keras.convert(
	sys.argv[1] + '.h5',
	input_names='image',
	image_input_names='image',
	image_scale=1/255.
)

coreml_model.author = 'mtbo.org'
coreml_model.license = 'mtbo.org'
coreml_model.short_description = "The Tiny YOLOv3 network"

print(coreml_model)

coreml_model.save(sys.argv[1] + '.mlmodel')
