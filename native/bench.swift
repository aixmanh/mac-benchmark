// bench: các bài test hiệu năng cho Apple Silicon.
// Giao thức đầu ra (server.py đọc):
//   PROGRESS <0..1> <mô tả>
//   RESULT {json}
//   ERROR <thông báo>
import Foundation
import Metal
import MetalPerformanceShaders
import AVFoundation
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

setvbuf(stdout, nil, _IOLBF, 0)

func now() -> Double {
    var t = timespec()
    clock_gettime(CLOCK_MONOTONIC_RAW, &t)
    return Double(t.tv_sec) + Double(t.tv_nsec) / 1e9
}

func progress(_ p: Double, _ msg: String) {
    print("PROGRESS \(String(format: "%.3f", min(max(p, 0), 1))) \(msg)")
}

func fail(_ msg: String) -> Never {
    print("ERROR \(msg)")
    exit(1)
}

/// big: số to hiển thị; unit: đơn vị; metric: giá trị số để lưu lịch sử (càng cao càng tốt).
func result(seconds: Double, big: String, unit: String, metric: Double, summary: String, details: [String], extra: [String: Any] = [:]) {
    var dict: [String: Any] = ["seconds": seconds, "big": big, "unit": unit, "metric": metric, "summary": summary, "details": details]
    dict.merge(extra) { $1 }
    let data = try! JSONSerialization.data(withJSONObject: dict)
    print("RESULT " + String(data: data, encoding: .utf8)!)
}

func fmt(_ v: Double, _ digits: Int = 1) -> String { String(format: "%.\(digits)f", v) }

let args = CommandLine.arguments
guard args.count >= 2 else { fail("usage: bench <cpu|gpu|draw|hevc|prores|memory|ssd> [option] [preview-dir]") }
let test = args[1]
let option = args.count > 2 ? args[2] : ""
let previewDir = args.count > 3 ? args[3] : NSTemporaryDirectory()
let coreCount = ProcessInfo.processInfo.activeProcessorCount

func metalDevice() -> (MTLDevice, MTLCommandQueue) {
    guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { fail("Không tìm thấy GPU Metal") }
    return (dev, q)
}

// MARK: - CPU (Mandelbrot)

func mandelRow(_ y: Int, w: Int, h: Int, maxIter: Int) -> Int {
    var total = 0
    let ci = Double(y) / Double(h) * 2.4 - 1.2
    for x in 0..<w {
        let cr = Double(x) / Double(w) * 3.2 - 2.2
        var zr = 0.0, zi = 0.0, i = 0
        while i < maxIter {
            let zr2 = zr * zr, zi2 = zi * zi
            if zr2 + zi2 > 4 { break }
            zi = 2 * zr * zi + ci
            zr = zr2 - zi2 + cr
            i += 1
        }
        total += i
    }
    return total
}

func runCPU(multi: Bool) {
    let w = 1600, maxIter = 2000
    let h = multi ? 1000 * 64 : 2000
    var rows = [Int](repeating: 0, count: h)
    progress(0.05, multi ? "Mandelbrot trên \(coreCount) luồng" : "Mandelbrot trên 1 luồng")
    let t0 = now()
    if multi {
        rows.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: h) { y in
                base[y] = mandelRow(y, w: w, h: h, maxIter: maxIter)
            }
        }
    } else {
        for y in 0..<h {
            rows[y] = mandelRow(y, w: w, h: h, maxIter: maxIter)
            if y % 50 == 0 { progress(Double(y) / Double(h), "Hàng \(y)/\(h)") }
        }
    }
    let dt = now() - t0
    let iters = Double(rows.reduce(0, +))
    let gips = iters / dt / 1e9
    result(seconds: dt, big: fmt(gips, 2), unit: "Giter/s",
           metric: gips,
           summary: multi ? "Đa nhân: \(coreCount) luồng, Mandelbrot \(w)×\(h)" : "Đơn nhân: Mandelbrot \(w)×\(h)",
           details: ["\(fmt(iters / 1e9, 1)) tỷ vòng lặp số phức (Double)", "\(fmt(Double(w * h) / 1e6, 1)) triệu điểm ảnh, tối đa \(maxIter) vòng/điểm"])
}

// MARK: - GPU (MPS matmul)

func runGPU(fp16: Bool) {
    let (dev, queue) = metalDevice()
    let n = 8192
    let type: MPSDataType = fp16 ? .float16 : .float32
    let rowBytes = MPSMatrixDescriptor.rowBytes(forColumns: n, dataType: type)
    let desc = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: rowBytes, dataType: type)
    let len = rowBytes * n
    progress(0.02, "Cấp phát 3 ma trận \(n)×\(n)")
    func makeFilled() -> MTLBuffer {
        let b = dev.makeBuffer(length: len, options: .storageModeShared)!
        let count = len / (fp16 ? 2 : 4)
        if fp16 {
            let p = b.contents().bindMemory(to: Float16.self, capacity: count)
            DispatchQueue.concurrentPerform(iterations: 64) { c in
                let s = count / 64
                for i in (c * s)..<((c + 1) * s) { p[i] = Float16(Float(i % 97) * 0.0005) }
            }
        } else {
            let p = b.contents().bindMemory(to: Float.self, capacity: count)
            DispatchQueue.concurrentPerform(iterations: 64) { c in
                let s = count / 64
                for i in (c * s)..<((c + 1) * s) { p[i] = Float(i % 97) * 0.0005 }
            }
        }
        return b
    }
    let a = MPSMatrix(buffer: makeFilled(), descriptor: desc)
    let b = MPSMatrix(buffer: makeFilled(), descriptor: desc)
    let c = MPSMatrix(buffer: dev.makeBuffer(length: len, options: .storageModePrivate)!, descriptor: desc)
    let mm = MPSMatrixMultiplication(device: dev, transposeLeft: false, transposeRight: false,
                                     resultRows: n, resultColumns: n, interiorColumns: n, alpha: 1, beta: 0)
    func batch(_ k: Int) -> Double {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<k { mm.encode(commandBuffer: cb, leftMatrix: a, rightMatrix: b, resultMatrix: c) }
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { fail("Lỗi GPU: \(e.localizedDescription)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }
    progress(0.1, "Khởi động GPU")
    _ = batch(2)
    var gpuTime = 0.0, count = 0
    let target = 4.0
    let t0 = now()
    while gpuTime < target {
        gpuTime += batch(5); count += 5
        progress(0.1 + 0.9 * gpuTime / target, "\(count) phép nhân ma trận")
    }
    let wall = now() - t0
    let flops = 2.0 * Double(n) * Double(n) * Double(n) * Double(count)
    let tflops = flops / gpuTime / 1e12
    result(seconds: wall, big: fmt(tflops, 1), unit: "TFLOPS", metric: tflops,
           summary: "Nhân ma trận \(n)×\(n) \(fp16 ? "FP16" : "FP32") bằng Metal Performance Shaders",
           details: ["\(count) phép nhân, \(fmt(flops / 1e12, 0)) nghìn tỷ phép tính", "GPU: \(dev.name)"])
}

// MARK: - DRAW (Metal render)

let drawShader = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float3 color; float2 uv; };
struct Uni { float time; float aspect; };

static float hash(uint n) {
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7fffffffU) / float(0x7fffffff);
}

vertex VOut vmain(uint vid [[vertex_id]], uint iid [[instance_id]], constant Uni &u [[buffer(0)]]) {
    const float2 corners[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,-1), float2(1,1), float2(-1,1) };
    float2 c = corners[vid];
    float r = hash(iid * 3u);
    float rad = 0.03 + 0.92 * pow(r, 0.7);
    float arm = floor(hash(iid * 3u + 1u) * 3.0) * 2.0943951;
    float spread = (hash(iid * 5u + 7u) - 0.5) * (0.5 + 1.2 * (1.0 - rad));
    float a = arm + rad * 5.0 + spread - u.time * 0.2 + 0.15 * sin(u.time * 0.5 + rad * 6.0);
    float2 center = float2(cos(a) * rad / u.aspect * 1.25, sin(a) * rad * 0.9);
    float size = 0.003 + 0.016 * hash(iid * 7u + 5u);
    VOut o;
    o.pos = float4(center + c * float2(size / u.aspect, size), 0, 1);
    o.color = mix(float3(1.0, 0.75, 0.35), float3(0.35, 0.55, 1.0), saturate(rad * 1.3)) + 0.25 * hash(iid * 11u);
    o.uv = c;
    return o;
}

fragment float4 fmain(VOut in [[stage_in]], constant Uni &u [[buffer(0)]]) {
    float d = length(in.uv);
    if (d > 1.0) discard_fragment();
    float acc = 0.0;
    for (int i = 0; i < 12; i++) {
        acc += sin(d * float(i + 1) * 3.1 + u.time * 1.7 + float(i)) / float(i + 1);
    }
    float glow = (1.0 - d) * (1.0 - d);
    float3 col = in.color * glow * (0.9 + 0.15 * acc);
    return float4(col, glow * 0.10);
}
"""

func runDraw(res: String) {
    let (dev, queue) = metalDevice()
    let (w, h): (Int, Int) = res == "8k" ? (7680, 4320) : res == "1080p" ? (1920, 1080) : (3840, 2160)
    let instances = 150_000
    let minSeconds = 6.0
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: drawShader, options: nil) } catch { fail("Lỗi biên dịch shader: \(error)") }
    let pd = MTLRenderPipelineDescriptor()
    pd.vertexFunction = lib.makeFunction(name: "vmain")
    pd.fragmentFunction = lib.makeFunction(name: "fmain")
    let ca = pd.colorAttachments[0]!
    ca.pixelFormat = .bgra8Unorm
    ca.isBlendingEnabled = true
    ca.sourceRGBBlendFactor = .sourceAlpha
    ca.destinationRGBBlendFactor = .one
    ca.sourceAlphaBlendFactor = .one
    ca.destinationAlphaBlendFactor = .one
    let pipeline: MTLRenderPipelineState
    do { pipeline = try dev.makeRenderPipelineState(descriptor: pd) } catch { fail("Lỗi pipeline: \(error)") }

    func makeTarget(_ w: Int, _ h: Int, _ mode: MTLStorageMode) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = mode
        return dev.makeTexture(descriptor: td)!
    }
    func encodeFrame(_ cb: MTLCommandBuffer, _ tex: MTLTexture, time: Float) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = tex
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].storeAction = .store
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.04, alpha: 1)
        let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
        var uni: (Float, Float) = (time, Float(tex.width) / Float(tex.height))
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBytes(&uni, length: 8, index: 0)
        enc.setFragmentBytes(&uni, length: 8, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: instances)
        enc.endEncoding()
    }

    let target = makeTarget(w, h, .private)
    progress(0.02, "Khởi động \(w)×\(h)")
    for i in 0..<10 { let cb = queue.makeCommandBuffer()!; encodeFrame(cb, target, time: Float(i) / 60); cb.commit(); if i == 9 { cb.waitUntilCompleted() } }

    let inflight = DispatchSemaphore(value: 3)
    let t0 = now()
    var last: MTLCommandBuffer?
    var frames = 0
    while now() - t0 < minSeconds {
        inflight.wait()
        let cb = queue.makeCommandBuffer()!
        encodeFrame(cb, target, time: Float(frames) / 60)
        cb.addCompletedHandler { _ in inflight.signal() }
        cb.commit()
        last = cb
        frames += 1
        if frames % 30 == 0 { progress((now() - t0) / minSeconds, "Khung hình \(frames)") }
    }
    last?.waitUntilCompleted()
    let dt = now() - t0
    let fps = Double(frames) / dt

    // Ảnh xem trước 1280×720
    var extra: [String: Any] = [:]
    let pv = makeTarget(1280, 720, .shared)
    let cb = queue.makeCommandBuffer()!
    encodeFrame(cb, pv, time: Float(frames) / 60)
    cb.commit(); cb.waitUntilCompleted()
    var bytes = [UInt8](repeating: 0, count: 1280 * 720 * 4)
    pv.getBytes(&bytes, bytesPerRow: 1280 * 4, from: MTLRegionMake2D(0, 0, 1280, 720), mipmapLevel: 0)
    let path = (previewDir as NSString).appendingPathComponent("preview_draw.png")
    if let provider = CGDataProvider(data: Data(bytes) as CFData),
       let img = CGImage(width: 1280, height: 720, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 1280 * 4,
                         space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
       let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(dest, img, nil)
        if CGImageDestinationFinalize(dest) { extra["preview"] = "preview_draw.png" }
    }
    let pixels = Double(w * h) * fps / 1e9
    result(seconds: dt, big: fmt(fps, 0), unit: "FPS", metric: fps,
           summary: "Render \(frames) khung hình \(w)×\(h), \(instances / 1000)k hạt phát sáng",
           details: ["\(fmt(pixels, 2)) Gpixel/giây", "Trung bình \(fmt(dt / Double(frames) * 1000, 2)) ms/khung hình"],
           extra: extra)
}

// MARK: - Video encode (HEVC / ProRes)

func makeFrames(w: Int, h: Int, count: Int, bgra: Bool) -> [CVPixelBuffer] {
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()]
    var out: [CVPixelBuffer] = []
    for f in 0..<count {
        var pbOpt: CVPixelBuffer?
        let fmt = bgra ? kCVPixelFormatType_32BGRA : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        CVPixelBufferCreate(nil, w, h, fmt, attrs as CFDictionary, &pbOpt)
        guard let pb = pbOpt else { fail("Không tạo được khung hình \(w)×\(h)") }
        CVPixelBufferLockBaseAddress(pb, [])
        // Họa tiết chuyển động + nhiễu để bộ mã hóa phải làm việc thật.
        @inline(__always) func noise(_ x: Int, _ y: Int) -> Int {
            var n = UInt32(truncatingIfNeeded: x &* 374761393 &+ y &* 668265263 &+ f &* 1442695041)
            n = (n ^ (n >> 13)) &* 1274126177
            return Int(n >> 27)
        }
        if bgra {
            let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            DispatchQueue.concurrentPerform(iterations: h) { y in
                let row = base + y * bpr
                for x in 0..<w {
                    let dx = x - w / 2 - Int(Double(w / 4) * sin(Double(f) * 0.2)), dy = y - h / 2
                    let ring = ((dx * dx + dy * dy) >> 10 + f * 6) & 255
                    row[x * 4] = UInt8(truncatingIfNeeded: ring + noise(x, y))
                    row[x * 4 + 1] = UInt8(truncatingIfNeeded: (x + f * 8) >> 2 + noise(y, x))
                    row[x * 4 + 2] = UInt8(truncatingIfNeeded: (y + f * 5) >> 2)
                    row[x * 4 + 3] = 255
                }
            }
        } else {
            let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self)
            let yBpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
            let cBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!.assumingMemoryBound(to: UInt8.self)
            let cBpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
            DispatchQueue.concurrentPerform(iterations: h) { y in
                let row = yBase + y * yBpr
                for x in 0..<w {
                    let dx = x - w / 2 - Int(Double(w / 4) * sin(Double(f) * 0.2)), dy = y - h / 2
                    let ring = ((dx * dx + dy * dy) >> 10 + f * 6) & 127
                    row[x] = UInt8(truncatingIfNeeded: 16 + ring + ((x + f * 8) >> 3) & 63 + noise(x, y))
                }
                if y < h / 2 {
                    let crow = cBase + y * cBpr
                    for x in 0..<(w / 2) {
                        crow[x * 2] = UInt8(truncatingIfNeeded: 128 + ((x + f * 3) >> 3) & 63 - 32)
                        crow[x * 2 + 1] = UInt8(truncatingIfNeeded: 128 + ((y + f * 2) >> 3) & 63 - 32)
                    }
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        out.append(pb)
    }
    return out
}

final class EncodeJob {
    let writer: AVAssetWriter
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputPixelBufferAdaptor
    let url: URL
    let queue = DispatchQueue(label: "encode")
    var appended = 0
    var finished = false

    init(codec: AVVideoCodecType, w: Int, h: Int, fps: Int32, bitrate: Int?, url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        self.url = url
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        var settings: [String: Any] = [AVVideoCodecKey: codec, AVVideoWidthKey: w, AVVideoHeightKey: h]
        if let br = bitrate {
            settings[AVVideoCompressionPropertiesKey] = [AVVideoAverageBitRateKey: br,
                                                         AVVideoExpectedSourceFrameRateKey: fps,
                                                         AVVideoMaxKeyFrameIntervalKey: Int(fps) * 2]
        }
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else { throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Codec không hỗ trợ"]) }
        writer.add(input)
    }

    func start(frames: [CVPixelBuffer], total: Int, fps: Int32, group: DispatchGroup) {
        group.enter()
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        input.requestMediaDataWhenReady(on: queue) { [self] in
            while input.isReadyForMoreMediaData && appended < total {
                if !adaptor.append(frames[appended % frames.count], withPresentationTime: CMTime(value: Int64(appended), timescale: fps)) {
                    break
                }
                appended += 1
            }
            if (appended >= total || writer.status == .failed) && !finished {
                finished = true
                input.markAsFinished()
                writer.finishWriting { group.leave() }
            }
        }
    }
}

func runEncode(prores: Bool, option: String) {
    let streams = option.contains("x4") ? 4 : 1
    let is8k = option.contains("8k")
    let (w, h) = is8k ? (7680, 4320) : (3840, 2160)
    let fps: Int32 = 60
    let total = is8k ? 300 : (prores && streams == 1 ? 1200 : 600)
    let codec: AVVideoCodecType
    let label: String
    var bitrate: Int? = nil
    if prores {
        if option.contains("4444") { codec = .proRes4444; label = "ProRes 4444" }
        else if option.contains("422hq") { codec = .proRes422HQ; label = "ProRes 422 HQ" }
        else { codec = .proRes422; label = "ProRes 422" }
    } else {
        codec = .hevc; label = "HEVC"
        bitrate = is8k ? 120_000_000 : 50_000_000
    }
    let bgra = codec == .proRes4444
    progress(0.01, "Tạo 30 khung hình nguồn \(w)×\(h)")
    let frames = makeFrames(w: w, h: h, count: 30, bgra: bgra)

    let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
    var jobs: [EncodeJob] = []
    do {
        for s in 0..<streams {
            jobs.append(try EncodeJob(codec: codec, w: w, h: h, fps: fps, bitrate: bitrate,
                                      url: tmp.appendingPathComponent("m5bench_\(test)_\(s).mov")))
        }
    } catch { fail("Không khởi tạo được bộ mã hóa \(label): \(error.localizedDescription)") }

    let group = DispatchGroup()
    let t0 = now()
    for j in jobs { j.start(frames: frames, total: total, fps: fps, group: group) }
    while group.wait(timeout: .now() + 0.5) == .timedOut {
        let done = jobs.reduce(0) { $0 + $1.appended }
        progress(0.05 + 0.95 * Double(done) / Double(total * streams), "Đã mã hóa \(done)/\(total * streams) khung hình")
    }
    let dt = now() - t0
    for j in jobs where j.writer.status != .completed {
        fail("Mã hóa thất bại: \(j.writer.error?.localizedDescription ?? "không rõ")")
    }
    let bytes = jobs.reduce(0) { acc, j in
        acc + ((try? FileManager.default.attributesOfItem(atPath: j.url.path)[.size] as? Int) ?? 0)
    }
    for j in jobs { try? FileManager.default.removeItem(at: j.url) }
    let encFps = Double(total * streams) / dt
    let videoSec = Double(total) / Double(fps)
    let realtime = videoSec * Double(streams) / dt
    result(seconds: dt, big: fmt(encFps, 0), unit: "FPS", metric: encFps,
           summary: "\(label) \(is8k ? "8K" : "4K")60\(streams > 1 ? " × \(streams) luồng song song" : ""): \(total * streams) khung hình",
           details: ["Nhanh gấp \(fmt(realtime, 1))× thời gian thực (\(fmt(videoSec * Double(streams), 0)) s video)",
                     "Dung lượng đầu ra \(fmt(Double(bytes) / 1e6, 0)) MB"])
}

// MARK: - Memory bandwidth

func runMemoryCPU() {
    let size = 4 << 30
    progress(0.02, "Cấp phát 2 × 4 GB")
    let src = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 1 << 14)
    let dst = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 1 << 14)
    defer { src.deallocate(); dst.deallocate() }
    let chunks = 256, chunk = size / chunks
    DispatchQueue.concurrentPerform(iterations: chunks) { c in
        memset(src + c * chunk, Int32(c & 0xff), chunk)
        memset(dst + c * chunk, 0, chunk)
    }
    var speeds: [Double] = [], totalTime = 0.0
    let passes = 24
    for p in 0..<passes {
        let t0 = now()
        DispatchQueue.concurrentPerform(iterations: chunks) { c in memcpy(dst + c * chunk, src + c * chunk, chunk) }
        let dt = now() - t0
        totalTime += dt
        speeds.append(2.0 * Double(size) / dt / 1e9)
        progress(Double(p + 1) / Double(passes), "Lượt \(p + 1)/\(passes): \(fmt(speeds.last!, 0)) GB/s")
    }
    let median = speeds.sorted()[passes / 2]
    result(seconds: totalTime, big: fmt(median, 0), unit: "GB/s", metric: median,
           summary: "Băng thông RAM từ CPU: memcpy 4 GB trên \(coreCount) luồng",
           details: ["Tính cả đọc + ghi, trung vị của \(passes) lượt", "Cao nhất \(fmt(speeds.max()!, 0)) GB/s, thấp nhất \(fmt(speeds.min()!, 0)) GB/s"])
}

let copyShader = """
#include <metal_stdlib>
using namespace metal;
kernel void copyk(device const float4 *src [[buffer(0)]], device float4 *dst [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}
"""

func runMemoryGPU() {
    let (dev, queue) = metalDevice()
    let size = 4 << 30
    let count = size / 16
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: copyShader, options: nil) } catch { fail("Lỗi shader: \(error)") }
    let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "copyk")!)
    progress(0.02, "Cấp phát 2 × 4 GB trên GPU")
    guard let src = dev.makeBuffer(length: size, options: .storageModePrivate),
          let dst = dev.makeBuffer(length: size, options: .storageModePrivate) else { fail("Không cấp phát được 8 GB cho GPU") }
    func run(_ k: Int) -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(ps)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        for _ in 0..<k {
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: ps.maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
        }
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        return cb.gpuEndTime - cb.gpuStartTime
    }
    _ = run(1)
    var best = 0.0, totalTime = 0.0
    let passes = 8
    for p in 0..<passes {
        let dt = run(4)
        totalTime += dt
        best = max(best, 2.0 * Double(size) * 4 / dt / 1e9)
        progress(Double(p + 1) / Double(passes), "Lượt \(p + 1)/\(passes): \(fmt(2.0 * Double(size) * 4 / dt / 1e9, 0)) GB/s")
    }
    result(seconds: totalTime, big: fmt(best, 0), unit: "GB/s", metric: best,
           summary: "Băng thông RAM từ GPU: sao chép 4 GB bằng compute shader",
           details: ["Tính cả đọc + ghi, lượt tốt nhất trong \(passes) lượt", "Trung bình \(fmt(2.0 * Double(size) * 4 * Double(passes) / totalTime / 1e9, 0)) GB/s"])
}

// MARK: - SSD

func runSSD(gb: Int) {
    let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("m5bench_ssd.tmp")
    let total = gb << 30
    let block = 64 << 20
    let threads = 4
    let perThread = total / threads
    let buf = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 1 << 14)
    defer { buf.deallocate(); unlink(path) }
    arc4random_buf(buf, block)

    progress(0.01, "Ghi \(gb) GB")
    var fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0o644)
    guard fd >= 0 else { fail("Không tạo được file tạm") }
    _ = fcntl(fd, F_NOCACHE, 1)
    var written = [Int](repeating: 0, count: threads)
    let tw = now()
    let wfd = fd
    DispatchQueue.global().async {
        while true {
            let done = written.reduce(0, +)
            if done >= total { break }
            progress(0.5 * Double(done) / Double(total), "Ghi \(fmt(Double(done) / 1073741824, 1))/\(gb) GB")
            usleep(300_000)
        }
    }
    DispatchQueue.concurrentPerform(iterations: threads) { t in
        var off = 0
        while off < perThread {
            let n = pwrite(wfd, buf, block, off_t(t * perThread + off))
            if n <= 0 { break }
            off += n
            written[t] = off
        }
    }
    _ = fcntl(fd, F_FULLFSYNC)
    let writeTime = now() - tw
    close(fd)

    progress(0.5, "Đọc \(gb) GB")
    fd = open(path, O_RDONLY)
    _ = fcntl(fd, F_NOCACHE, 1)
    let rfd = fd
    var readBytes = [Int](repeating: 0, count: threads)
    let tr = now()
    DispatchQueue.concurrentPerform(iterations: threads) { t in
        let rb = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 1 << 14)
        defer { rb.deallocate() }
        var off = 0
        while off < perThread {
            let n = pread(rfd, rb, block, off_t(t * perThread + off))
            if n <= 0 { break }
            off += n
            readBytes[t] = off
            if t == 0 { progress(0.5 + 0.5 * Double(off) / Double(perThread), "Đọc \(gb) GB") }
        }
    }
    let readTime = now() - tr
    close(fd)
    let w = Double(written.reduce(0, +)) / writeTime / 1e9
    let r = Double(readBytes.reduce(0, +)) / readTime / 1e9
    result(seconds: writeTime + readTime, big: "\(fmt(w, 1)) / \(fmt(r, 1))", unit: "GB/s ghi / đọc", metric: (w + r) / 2,
           summary: "SSD nội bộ: ghi rồi đọc tuần tự \(gb) GB (bỏ qua bộ nhớ đệm)",
           details: ["Ghi \(fmt(w, 2)) GB/s trong \(fmt(writeTime, 1)) s", "Đọc \(fmt(r, 2)) GB/s trong \(fmt(readTime, 1)) s"])
}

// MARK: - main

switch test {
case "cpu": runCPU(multi: option != "single")
case "gpu": runGPU(fp16: option == "fp16")
case "draw": runDraw(res: option.isEmpty ? "4k" : option)
case "hevc": runEncode(prores: false, option: option.isEmpty ? "4k" : option)
case "prores": runEncode(prores: true, option: option.isEmpty ? "422hq" : option)
case "memory": option == "gpu" ? runMemoryGPU() : runMemoryCPU()
case "ssd": runSSD(gb: Int(option) ?? 16)
default: fail("Bài test không tồn tại: \(test)")
}
