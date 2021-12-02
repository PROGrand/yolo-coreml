//
//  YOLO.swift
//  yolov4
//
//  Copyright © 2021 Vladimir E. Koltunov. All rights reserved.
//

import Foundation
import UIKit
import CoreML

class YOLO {
	public let inputWidth: Int
	public let inputHeight: Int
	public let shapePoint: Int
	
	var model : yolov4?
	
	// Tweak these values to get more or fewer predictions.
	public let confidenceThreshold: Float = 0.016
	public let iouThreshold: Float = 0.1
	public let maxBoundingBoxes: Int = 5
	
	struct Prediction {
		let classIndex: Int
		let score: Float
		let rect: CGRect
	}
	
	
	
	public init(width: Int, height: Int, channels: Int) throws {
		inputWidth = width
		inputHeight = height
		
		shapePoint = inputWidth * inputHeight * channels
		
		weak var welf = self
		yolov4.load(completionHandler: { result in
			switch result {
			case .failure(let error):
				print(error)
			case .success(let modelSuccess):
				welf?.model = modelSuccess
			}
		})
		
	}
	
	public func predict(image: CVPixelBuffer) throws -> [Prediction] {
		
		let startTime = Date.now
		if let output = try? model?.prediction(input_1: image) {
			
			print("Inference time: \(Date.now.timeIntervalSince(startTime))")
			
			let comp = computeBoundingBoxes(features: output)
			
			print("Prediction time: \(Date.now.timeIntervalSince(startTime))")
			return comp
		} else {
			return []
		}
		
	}
	
	struct Output {
		var name: String
		var array: MLMultiArray
		var rows: Int
		var cols: Int
		var blockSize: Int
	}
	
	public func computeBoundingBoxes(features: yolov4Output) -> [Prediction] {
		
		var predictions = [Prediction]()
		
		let featureNames = features.featureNames
		
		let outputFeatures = featureNames.map { name in
			(name, features.featureValue(for: name)!.multiArrayValue!)}.map { pair in
				Output(name: pair.0, array: pair.1, rows: pair.1.shape[1].intValue, cols: pair.1.shape[2].intValue, blockSize: pair.1.shape[3].intValue)
			}.sorted { $0.rows > $1.rows}
		
		
		let numClasses = labels.count
		
		var index = 0
		for output in outputFeatures {
			predictions.append(contentsOf:self.computeBoundingBoxes(output: output,	numClasses: numClasses, anchors: anchors[index]) )
			index += 1
		}
		
		return nonMaxSuppression(boxes: predictions, limit: maxBoundingBoxes, threshold: iouThreshold)
	}
	
	public func computeBoundingBoxes(output: Output, numClasses:Int, anchors: [Float]) -> [Prediction] {
		let boxesPerCell = output.blockSize / (numClasses + 5)
		let cellHeight = inputHeight / output.rows
		let cellWidth = inputWidth / output.cols
		
		let cnt = output.array.count
		let cnt_req = output.blockSize * output.rows * output.cols
		assert(cnt == cnt_req)
		assert(output.array.dataType == .float32)
		
		
		var predictions = [Prediction]()
		
		let featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(output.array.dataPointer))
		let yStride = output.array.strides[1].intValue
		let xStride = output.array.strides[2].intValue
		let channelStride = output.array.strides[3].intValue
		
		@inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
			return y * yStride + x * xStride + channel * channelStride
		}
		
		
		var confidenceMax = Float(0)
		
		for cy in 0 ..< output.rows {
			for cx in 0 ..< output.cols {
				for b in 0 ..< boxesPerCell {
					let channel = b * (numClasses + 5)
					
					// The fast way:
					let tx = Float(featurePointer[offset(channel    , cx, cy)])
					let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
					let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
					let th = Float(featurePointer[offset(channel + 3, cx, cy)])
					let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
					
					let x = (Float(cx) + sigmoid(tx)) * Float(cellHeight)
					let y = (Float(cy) + sigmoid(ty)) * Float(cellWidth)
					
					let w = exp(tw) * anchors[2 * b    ]
					let h = exp(th) * anchors[2 * b + 1]
					
					let confidence = sigmoid(tc)
					
					var classes = [Float](repeating: 0, count: numClasses)
					
					
					for c in 0 ..< numClasses {
						classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
					}
					
					classes = softmax(classes)
					
					let (detectedClass, bestClassScore) = classes.argmax()
					
					let confidenceInClass = bestClassScore * confidence
					
					confidenceMax = max(confidenceMax, confidenceInClass)
					
					if confidenceInClass > confidenceThreshold {
						let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
										  width: CGFloat(w), height: CGFloat(h))
						
						let prediction = Prediction(classIndex: detectedClass,
													score: confidenceInClass,
													rect: rect)
						predictions.append(prediction)
					}
				}
			}
		}
		
		//print("predictions [\(output.rows) x \(output.cols)]: max conf: \(confidenceMax)")
		
		// We already filtered out any bounding boxes that have very low scores,
		// but there still may be boxes that overlap too much with others. We'll
		// use "non-maximum suppression" to prune those duplicate bounding boxes.
		return nonMaxSuppression(boxes: predictions, limit: maxBoundingBoxes, threshold: iouThreshold)
	}
	
}

