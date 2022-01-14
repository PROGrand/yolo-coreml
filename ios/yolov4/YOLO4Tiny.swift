//
//  YOLO4Tiny.swift
//  yolov4
//
//  Created by Vladimir E. Koltunov on 18.12.2021.
//  Copyright © 2021 mtbo.org. All rights reserved.
//

import Foundation
import CoreML
//
//class YOLO4Tiny : YOLOWrapper {
//
//	override func load(width: Int, height: Int, confidence: Float, nms: Float, maxBoundingBoxes: Int) async throws {
//		let model = try await yolov4_tiny.load(contentsOf: Bundle.main.url(forResource: "yolov4-tiny", withExtension: "mlmodelc")!).model
//		try yolo = YOLO(width: width, height: height,
//				channels: 3, model: model, confidenceThreshold: confidence, nmsThreshold: nms, maxBoundingBoxes: maxBoundingBoxes)
//	}
//}


class YOLO4My : YOLOWrapper {

	override func load(width: Int, height: Int, confidence: Float, nms: Float, maxBoundingBoxes: Int) async throws {
		print("YOLO load")
		
		let model = try await my.load(contentsOf: Bundle.main.url(forResource: "my", withExtension: "mlmodelc")!).model
		
		try yolo = YOLO(width: width, height: height,
				channels: 3, model: model, confidenceThreshold: confidence, nmsThreshold: nms, maxBoundingBoxes: maxBoundingBoxes)
		print("YOLO loaded")
	}
}
