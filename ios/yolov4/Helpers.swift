//
//  Helpers.swift
//  yolov4
//
//  Copyright © 2021 Vladimir E. Koltunov. All rights reserved.
//

import Foundation
import UIKit
import CoreML
import Accelerate

/**
 Computes intersection-over-union overlap between two bounding boxes.
 */
public func IOU(a: CGRect, b: CGRect) -> Float {
	let areaA = a.width * a.height
	if areaA <= 0 { return 0 }
	
	let areaB = b.width * b.height
	if areaB <= 0 { return 0 }
	
	let intersectionMinX = max(a.minX, b.minX)
	let intersectionMinY = max(a.minY, b.minY)
	let intersectionMaxX = min(a.maxX, b.maxX)
	let intersectionMaxY = min(a.maxY, b.maxY)
	let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
	max(intersectionMaxX - intersectionMinX, 0)
	return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
	/**
	 Returns the index and value of the largest element in the array.
	 */
	public func argmax() -> (Int, Element) {
		precondition(self.count > 0)
		var maxIndex = 0
		var maxValue = self[0]
		for i in 1..<self.count {
			if self[i] > maxValue {
				maxValue = self[i]
				maxIndex = i
			}
		}
		return (maxIndex, maxValue)
	}
}

/**
 Logistic sigmoid.
 */
public func sigmoid(_ x: Float) -> Float {
	return 1 / (1 + exp(-x))
}

/**
 Computes the "softmax" function over an array.
 
 Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
 
 This is what softmax looks like in "pseudocode" (actually using Python
 and numpy):
 
 x -= np.max(x)
 exp_scores = np.exp(x)
 softmax = exp_scores / np.sum(exp_scores)
 
 First we shift the values of x so that the highest value in the array is 0.
 This ensures numerical stability with the exponents, so they don't blow up.
 */
public func softmax(_ x: [Float]) -> [Float] {
	var x = x
	let len = vDSP_Length(x.count)
	
	// Find the maximum value in the input array.
	var max: Float = 0
	vDSP_maxv(x, 1, &max, len)
	
	// Subtract the maximum from all the elements in the array.
	// Now the highest value in the array is 0.
	max = -max
	vDSP_vsadd(x, 1, &max, &x, 1, len)
	
	// Exponentiate all the elements in the array.
	var count = Int32(x.count)
	vvexpf(&x, x, &count)
	
	// Compute the sum of all exponentiated values.
	var sum: Float = 0
	vDSP_sve(x, 1, &sum, len)
	
	// Divide each element by the sum. This normalizes the array contents
	// so that they all add up to 1.
	vDSP_vsdiv(x, 1, &sum, &x, 1, len)
	
	return x
}
