// face: tìm khuôn mặt lớn nhất trong ảnh bằng Vision, thử 3 hướng (thẳng, xoay phải, xoay trái).
// Cách dùng: face <ảnh.png>
// In JSON: {"orientation": "up|right|left", "x":..,"y":..,"w":..,"h":..} — toạ độ chuẩn hoá trong ảnh
// đã xoay thẳng, gốc ở góc trên trái; hoặc {"orientation": null} nếu không thấy mặt.
//   right = ảnh cần xoay 90° theo chiều kim đồng hồ để thẳng; left = xoay 90° ngược chiều.
import Foundation
import Vision
import ImageIO

guard CommandLine.arguments.count > 1,
      let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: CommandLine.arguments[1]) as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    print("{\"orientation\": null}")
    exit(0)
}

// Vision tìm được cả mặt nghiêng/lộn ngược, nên chọn hướng làm mặt gần thẳng đứng nhất (|roll| nhỏ nhất).
var best: (name: String, box: CGRect, score: Double)? = nil
for (name, orientation) in [("up", CGImagePropertyOrientation.up), ("right", .right), ("left", .left)] {
    let request = VNDetectFaceRectanglesRequest()
    request.revision = VNDetectFaceRectanglesRequestRevision3
    try? VNImageRequestHandler(cgImage: image, orientation: orientation).perform([request])
    for face in request.results ?? [] {
        let roll = abs(face.roll?.doubleValue ?? 0)
        let area = Double(face.boundingBox.width * face.boundingBox.height)
        let score = -roll + area  // roll tính bằng radian (0 = thẳng, π = lộn ngược)
        if best == nil || score > best!.score {
            best = (name, face.boundingBox, score)
        }
    }
}

if let b = best {
    // Vision dùng gốc dưới trái -> đổi sang gốc trên trái
    let y = 1 - b.box.origin.y - b.box.height
    print(String(format: "{\"orientation\": \"%@\", \"x\": %.4f, \"y\": %.4f, \"w\": %.4f, \"h\": %.4f}",
                 b.name, b.box.origin.x, y, b.box.width, b.box.height))
} else {
    print("{\"orientation\": null}")
}
