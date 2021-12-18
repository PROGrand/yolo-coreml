//
//  YOLOWrapper.swift
//  yolov4
//
//  Copyright © 2021 Vladimir E. Koltunov. All rights reserved.
//

import Foundation
import UIKit
import CoreML

class YOLOWrapper {
	
	var yolo : YOLO?
	
	@objc
	func load(width: Int, height: Int, confidence: Float, nms: Float, maxBoundingBoxes: Int) async throws {
	}
	
	@objc
	public func predict(image: CVPixelBuffer) throws -> [Prediction] {
		return try yolo?.predict(image: image) ?? []
	}
	
	var names: [Int: String] {
		get {
			return yolo?.names ?? [:]
		}
	}
}

@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
class imageInput : MLFeatureProvider {

	var input: CVPixelBuffer

	var featureNames: Set<String> {
		get {
			return ["input_1"]
		}
	}
	
	func featureValue(for featureName: String) -> MLFeatureValue? {
		if (featureName == "input_1") {
			return MLFeatureValue(pixelBuffer: input)
		}
		return nil
	}
	
	init(_ input: CVPixelBuffer) {
		self.input = input
	}
}

@objc class Prediction : NSObject {
	@objc let classIndex: Int
	@objc let score: Float
	@objc let rect: CGRect
	
	public init(classIndex: Int,
	 score: Float,
	 rect: CGRect) {
		self.classIndex = classIndex
		self.score = score
		self.rect = rect
	}
}



@objc
class YOLO : NSObject {
	public let inputWidth: Int
	public let inputHeight: Int
	public let shapePoint: Int
	
	var model : MLModel?
	
	// Tweak these values to get more or fewer predictions.
	public let confidenceThreshold: Float
	public let iouThreshold: Float
	public let maxBoundingBoxes: Int
	public let anchors: [(Float,Float)]
	public let names: [Int: String]
	public let classesCount : Int
	public let newCoords: Bool
	
	
	public init(width: Int, height: Int, channels: Int, model: MLModel, confidenceThreshold: Float, nmsThreshold: Float, maxBoundingBoxes: Int, newCoords: Bool = false) throws {
		self.confidenceThreshold = confidenceThreshold
		self.anchors = YOLO.parseAnchors(model: model)
		self.names = try YOLO.parseNames(model: model)
		self.classesCount = names.count
		self.iouThreshold = nmsThreshold
		self.maxBoundingBoxes = maxBoundingBoxes
		inputWidth = width
		inputHeight = height
		self.newCoords = newCoords
		
		shapePoint = inputWidth * inputHeight * channels
		
		self.model = model
	}
	
	public static func parseAnchors(model: MLModel) -> [(Float,Float)] {
		let userDefines = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? NSDictionary
		let anchorsString = userDefines?["yolo.anchors"] as? String
		
		return YOLO.parseAnchorsString(anchorsString: anchorsString!)
	}
	
	public static func parseNames(model: MLModel) throws -> [Int:String] {
		guard let userDefines = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? NSDictionary else { return [:]}
		guard let anchorsString = userDefines["yolo.names"] as? String else { return [:] }
		guard let data = anchorsString.data(using: .utf8) else { return [:] }
		return try JSONDecoder().decode([Int:String].self, from: data)
	}
	
	static func scaleAnchors(_ a: [Float], _ width: Int, _ height: Int) -> [(Float, Float)] {
		var anchors = [(Float, Float)](repeating: (0.0, 0.0), count: a.count / 2)
		for n in 0 ..< a.count / 2 {
			anchors[n].0 = a[2 * n] / Float(width)
			anchors[n].1 = a[2 * n + 1] / Float(height)
		}
		
		return anchors
	}
	
	static func parseAnchorsString(anchorsString: String) -> [(Float, Float)] {
		let splitted = anchorsString.trimmingCharacters(in: ["[","]", " "]).split(separator: "\n")
		
		var anchors = [(Float, Float)](repeating: (0.0, 0.0), count: splitted.count)
		for n in 0 ..< splitted.count {
			let pair = splitted[n].description.trimmingCharacters(in: ["[", "]", ",", " "]).split(separator: ",").map { val in
				Float(val.trimmingCharacters(in: [" "]))
			}
			anchors[n].0 = pair[0]!
			anchors[n].1 = pair[1]!
		}
		
		return anchors
	}
	
	
	@objc
	public func predict(image: CVPixelBuffer) throws -> [Prediction] {
		
		let startTime = Date.now
		
		let input = imageInput(image)
		
		if let output = try? model?.prediction(from: input) {
			
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
	
	public func computeBoundingBoxes(features: MLFeatureProvider) -> [Prediction] {
		
		var predictions = [Prediction]()
		
		let featureNames = features.featureNames
		
		let outputFeatures = featureNames.map { name in
			(name, features.featureValue(for: name)!.multiArrayValue!)}.map { pair in
				Output(name: pair.0, array: pair.1, rows: pair.1.shape[1].intValue, cols: pair.1.shape[2].intValue, blockSize: pair.1.shape[3].intValue)
			}.sorted { $0.rows > $1.rows}
		
		
		var index = 0
		let anchorStride = anchors.count / outputFeatures.count
		
		for output in outputFeatures {
			let _anchors = Array<(Float, Float)>(anchors[index * anchorStride ..< (index+1) * anchorStride])
			predictions.append(contentsOf:self.computeBoundingBoxes(output: output,	anchors: _anchors) )
			index += 1
		}
		
		return nonMaxSuppression(boxes: predictions, limit: maxBoundingBoxes, threshold: iouThreshold)
	}
	
	func computeBoundingBoxes(output: Output, anchors: [(Float, Float)]) -> [Prediction] {
		
		let boxesPerCell = output.blockSize / (classesCount + 5)
		
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
		
		for y in 0 ..< output.rows {
			for x in 0 ..< output.cols {
				for b in 0 ..< boxesPerCell {
					let channel = b * (classesCount + 5)
					
					// The fast way:
					var bbox_0 = Float(featurePointer[offset(channel    , x, y)])
					var bbox_1 = Float(featurePointer[offset(channel + 1, x, y)])
					var bbox_2 = Float(featurePointer[offset(channel + 2, x, y)])
					var bbox_3 = Float(featurePointer[offset(channel + 3, x, y)])
					let obj = Float(featurePointer[offset(channel + 4, x, y)])
		  
					var exist = false
					var classes = [Float](repeating: 0, count: classesCount)
					
					if (obj > confidenceThreshold) {
						
						for c in 0 ..< classesCount {
							let bbox_c = Float(featurePointer[offset(channel + 5 + c, x, y)])
						   
							let prob = bbox_c * obj
							if (prob > confidenceThreshold) {
								classes[c] = prob;
								exist   = true
							} else {
								classes[c] = 0
							}
						}

					}
					
					let (detectedClass, bestClassScore) = classes.argmax()
										
					let confidenceInClass = bestClassScore * obj
					
					confidenceMax = max(confidenceMax, confidenceInClass)
					
					if (exist) {
						bbox_0 = (bbox_0 + Float(x)) / Float(output.cols)
						bbox_1 = (bbox_1 + Float(y)) / Float(output.rows)
						let a = anchors[b]
						
						if (newCoords) {
							bbox_2 = bbox_2 * bbox_2 * 4 * a.0
							bbox_3 = bbox_3 * bbox_3 * 4 * a.1
						} else {
							
							bbox_2 = exp(bbox_2) * a.0
							bbox_3 = exp(bbox_3) * a.1
						}
						
						
						let rect = CGRect(x: CGFloat(bbox_0 - bbox_2/2.0), y: CGFloat(bbox_1 - bbox_3/2.0),
										  width: CGFloat(bbox_2), height: CGFloat(bbox_3))

						let prediction = Prediction(classIndex: detectedClass,
													score: confidenceInClass,
													rect: rect)
						predictions.append(prediction)
					}
					
					

				}
			}
		}
		
		print("predictions [\(output.rows) x \(output.cols)]: max conf: \(confidenceMax), threshold: \(confidenceThreshold)")
		
		// We already filtered out any bounding boxes that have very low scores,
		// but there still may be boxes that overlap too much with others. We'll
		// use "non-maximum suppression" to prune those duplicate bounding boxes.
		return nonMaxSuppression(boxes: predictions, limit: maxBoundingBoxes, threshold: iouThreshold)
	}
	
	/**
	 Removes bounding boxes that overlap too much with other boxes that have
	 a higher score.
	 
	 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
	 
	 - Parameters:
	 - boxes: an array of bounding boxes and their scores
	 - limit: the maximum number of boxes that will be selected
	 - threshold: used to decide whether boxes overlap too much
	 */
	func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {
		
		// Do an argsort on the confidence scores, from high to low.
		let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
		
		var selected: [Prediction] = []
		var active = [Bool](repeating: true, count: boxes.count)
		var numActive = active.count
		
		// The algorithm is simple: Start with the box that has the highest score.
		// Remove any remaining boxes that overlap it more than the given threshold
		// amount. If there are any boxes left (i.e. these did not overlap with any
		// previous boxes), then repeat this procedure, until no more boxes remain
		// or the limit has been reached.
		outer: for i in 0..<boxes.count {
			if active[i] {
				let boxA = boxes[sortedIndices[i]]
				selected.append(boxA)
				if selected.count >= limit { break }
				
				for j in i+1..<boxes.count {
					if active[j] {
						let boxB = boxes[sortedIndices[j]]
						if IOU(a: boxA.rect, b: boxB.rect) > threshold {
							active[j] = false
							numActive -= 1
							if numActive <= 0 { break outer }
						}
					}
				}
			}
		}
		return selected
	}
}

