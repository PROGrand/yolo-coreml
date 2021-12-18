//
//  ViewController.swift
//  yolov4
//
//  Copyright © 2021 Vladimir E. Koltunov. All rights reserved.
//

import UIKit
import Vision
import AVFoundation
import CoreMedia
import VideoToolbox

class ViewController: UIViewController {
	
	let inputWidth = 416
	let inputHeight = 416
	let maxBoundingBoxes = 10
	let labelHeight:CGFloat = 50.0
	
	var yolo = YOLO4Tiny()
	
	var videoCapture: VideoCapture!
	var request: VNCoreMLRequest!
	var startTimes: [CFTimeInterval] = []
	
	var boundingBoxes = [BoundingBox]()
	var colors: [UIColor] = []
	
	let ciContext = CIContext()
	var resizedPixelBuffer: CVPixelBuffer?
	
	var framesDone = 0
	var frameCapturingStartTime = CACurrentMediaTime()
	let semaphore = DispatchSemaphore(value: 2)
	
	
	let timeLabel: UILabel = {
		let label = UILabel()
		return label
	}()
	
	let  videoPreview: UIView = {
		let view = UIView()
		return view
	}()
	
	override func viewDidLoad() {
		super.viewDidLoad()
		
		Task { [weak self] in
			print("TASK IN")
			try! await self?.yolo.load(width: inputWidth, height: inputHeight, confidence: 0.4, nms: 0.6, maxBoundingBoxes: maxBoundingBoxes)
			print("TASK OUT")
		}

		timeLabel.frame = CGRect(x: 0, y: UIScreen.main.bounds.size.height - self.labelHeight, width: UIScreen.main.bounds.size.width, height: self.labelHeight)
		let frm = self.view.frame
		
		videoPreview.frame = frm
		
		view.addSubview(timeLabel)
		view.addSubview(videoPreview)
		
		timeLabel.text = ""
		
		setUpBoundingBoxes()
		setUpCoreImage()
		setUpCamera()
		
		frameCapturingStartTime = CACurrentMediaTime()
	}
	
	
	func setUpBoundingBoxes() {
		for _ in 0 ..< maxBoundingBoxes {
			boundingBoxes.append(BoundingBox())
		}
		
		// Make colors for the bounding boxes. There is one color for each class,
		// 20 classes in total.
		for r: CGFloat in [0.1,0.2, 0.3,0.4,0.5, 0.6,0.7, 0.8,0.9, 1.0] {
			for g: CGFloat in [0.3,0.5, 0.7,0.9] {
				for b: CGFloat in [0.4,0.6 ,0.8] {
					let color = UIColor(red: r, green: g, blue: b, alpha: 1)
					colors.append(color)
				}
			}
		}
	}
	
	func setUpCoreImage() {
		let status = CVPixelBufferCreate(nil, inputWidth, inputHeight,
										 kCVPixelFormatType_32BGRA, nil,
										 &resizedPixelBuffer)
		if status != kCVReturnSuccess {
			print("Error: could not create resized pixel buffer", status)
		}
	}
	
	func setUpCamera() {
		videoCapture = VideoCapture()
		videoCapture.delegate = self
		videoCapture.fps = 25
		
		videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { [weak self] success in
			if success {
				// Add the video preview into the UI.
				if let previewLayer = self?.videoCapture.previewLayer {
					self?.videoPreview.layer.addSublayer(previewLayer)
					self?.resizePreviewLayer()
				}
				
				
				// Add the bounding box layers to the UI, on top of the video preview.
				DispatchQueue.main.async {
					guard let  boxes = self?.boundingBoxes,let videoLayer  = self?.videoPreview.layer else {return}
					for box in boxes {
						box.addToLayer(videoLayer)
					}
					self?.semaphore.signal()
				}
				
				
				// Once everything is set up, we can start capturing live video.
				self?.videoCapture.start()

			}
		}
	}
	
	override var preferredStatusBarStyle: UIStatusBarStyle {
		return .lightContent
	}
	
	override func viewWillLayoutSubviews() {
		super.viewWillLayoutSubviews()
		resizePreviewLayer()
	}
	
	func resizePreviewLayer() {
		videoCapture.previewLayer?.frame = videoPreview.bounds
	}
	
	
	func predict(pixelBuffer: CVPixelBuffer) {
		
		// Measure how long it takes to predict a single video frame.
		let startTime = CACurrentMediaTime()
		
		// Resize the input with Core Image.
		guard let resizedPixelBuffer = resizedPixelBuffer else { return }
		let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
		let sx = CGFloat(inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
		let sy = CGFloat(inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
		let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
		let scaledImage = ciImage.transformed(by: scaleTransform)
		ciContext.render(scaledImage, to: resizedPixelBuffer)
		
		
		if let boundingBoxes = try? yolo.predict(image: resizedPixelBuffer) {
			let elapsed = CACurrentMediaTime() - startTime
			showOnMainThread(boundingBoxes, elapsed)
		}
	}
	
	
	func showOnMainThread(_ boundingBoxes: [Prediction], _ elapsed: CFTimeInterval) {
		DispatchQueue.main.async { [weak self] in
			// For debugging, to make sure the resized CVPixelBuffer is correct.
			//var debugImage: CGImage?
			//VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
			//self.debugImageView.image = UIImage(cgImage: debugImage!)
			
			self?.show(predictions: boundingBoxes)
			
			guard  let fps = self?.measureFPS() else{return}
			self?.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
			
			self?.semaphore.signal()
		}
	}
	
	func show(predictions: [Prediction]) {
		for i in 0..<boundingBoxes.count {
			if i < predictions.count {
				let prediction = predictions[i]
				
				let width = view.bounds.width
				let height = width * 1280 / 720
				let scaleX = width
				let scaleY = height
				let top = (view.bounds.height - height) / 2
				
				// Translate and scale the rectangle to our own coordinate system.
				var rect = prediction.rect
				rect.origin.x *= scaleX
				rect.origin.y *= scaleY
				rect.origin.y += top
				rect.size.width *= scaleX
				rect.size.height *= scaleY
				
				// Show the bounding box.
				let label = String(format: "%@ %.1f", yolo.names[prediction.classIndex] ?? "<unknown>", prediction.score)
				let color = colors[prediction.classIndex]
				boundingBoxes[i].show(frame: rect, label: label, color: color)
			} else {
				boundingBoxes[i].hide()
			}
		}
	}
	
	func measureFPS() -> Double {
		// Measure how many frames were actually delivered per second.
		framesDone += 1
		let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
		let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
		if frameCapturingElapsed > 1 {
			framesDone = 0
			frameCapturingStartTime = CACurrentMediaTime()
		}
		return currentFPSDelivered
	}
	
	
	override func didReceiveMemoryWarning() {
		super.didReceiveMemoryWarning()
		// Dispose of any resources that can be recreated.
	}
	
	
}


extension ViewController: VideoCaptureDelegate {
	func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
		if let pixelBuffer = pixelBuffer {
			DispatchQueue.global().async { [weak self] in
				self?.predict(pixelBuffer: pixelBuffer)
			}
		}
	}
}
