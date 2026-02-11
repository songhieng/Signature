#!/usr/bin/env python3
import argparse
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import openvino as ov
except ModuleNotFoundError:
    try:
        import openvino.runtime as ov  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenVINO Python package is not installed. Install with: pip install openvino"
        ) from exc


def parse_device_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def canonical_device_name(core: ov.Core, preferred: str) -> Optional[str]:
    pref = preferred.strip()
    if not pref:
        return None

    available = core.available_devices
    if pref in available:
        return pref

    pref_upper = pref.upper()
    if pref_upper == "AUTO" or pref_upper.startswith("AUTO:"):
        return pref

    for dev in available:
        base = dev.split(".", 1)[0].upper()
        if base == pref_upper:
            return dev
    return None


def pick_device(core: ov.Core, requested: str, fallback: List[str]) -> str:
    if fallback:
        for item in fallback:
            candidate = canonical_device_name(core, item)
            if candidate:
                return candidate
        raise RuntimeError(
            f"None of fallback devices are available: {fallback}. "
            f"Detected devices: {core.available_devices}"
        )

    candidate = canonical_device_name(core, requested)
    if candidate:
        return candidate
    return requested


@dataclass
class InputSpec:
    name: str
    shape: List[int]
    layout: str
    height: int
    width: int


def resolve_input_spec(model: ov.Model, fallback_size: Tuple[int, int]) -> InputSpec:
    def dim_is_dynamic(dim: object) -> bool:
        flag = getattr(dim, "is_dynamic", None)
        if callable(flag):
            return bool(flag())
        if flag is not None:
            return bool(flag)
        # Fallback for variants exposing only is_static.
        static_flag = getattr(dim, "is_static", None)
        if callable(static_flag):
            return not bool(static_flag())
        if static_flag is not None:
            return not bool(static_flag)
        return False

    def dim_to_int(dim: object) -> int:
        # Most OpenVINO builds expose get_length() for static dimensions.
        get_length = getattr(dim, "get_length", None)
        if callable(get_length):
            return int(get_length())
        # Fallback for builds where Dimension is directly castable.
        return int(dim)  # type: ignore[arg-type]

    input_port = model.input(0)
    shape = input_port.partial_shape
    rank = len(shape)
    if rank != 4:
        raise RuntimeError(
            f"Only 4D model input is supported in this script. Got rank={rank}."
        )

    concrete_shape: List[int] = []
    for idx, dim in enumerate(shape):
        if dim_is_dynamic(dim):
            if idx == 0:
                concrete_shape.append(1)
            elif idx == 1:
                concrete_shape.append(3)
            elif idx == 2:
                concrete_shape.append(fallback_size[1])
            else:
                concrete_shape.append(fallback_size[0])
        else:
            concrete_shape.append(dim_to_int(dim))

    # Guess layout from shape.
    if concrete_shape[1] in (1, 3):
        layout = "NCHW"
        h, w = concrete_shape[2], concrete_shape[3]
    elif concrete_shape[3] in (1, 3):
        layout = "NHWC"
        h, w = concrete_shape[1], concrete_shape[2]
    else:
        # Fallback to NCHW conventions.
        layout = "NCHW"
        h, w = concrete_shape[2], concrete_shape[3]

    return InputSpec(
        name=input_port.get_any_name(),
        shape=concrete_shape,
        layout=layout,
        height=h,
        width=w,
    )


def preprocess_frame(frame: np.ndarray, spec: InputSpec) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    resized = cv2.resize(frame, (spec.width, spec.height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    data = rgb.astype(np.float32)

    if spec.layout == "NCHW":
        data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    return data


def extract_depth_map(output: np.ndarray) -> np.ndarray:
    data = np.array(output)
    data = np.squeeze(data)

    if data.ndim == 3:
        if data.shape[0] in (1, 3):
            data = data[0]
        elif data.shape[2] in (1, 3):
            data = data[..., 0]
        else:
            data = data.mean(axis=0)
    if data.ndim != 2:
        raise RuntimeError(f"Unexpected output shape for depth visualization: {output.shape}")

    depth = data.astype(np.float32)
    min_v, max_v = float(depth.min()), float(depth.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(depth, dtype=np.uint8)

    depth_norm = ((depth - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    return depth_norm


def infer_depth_map(infer_request: ov.InferRequest, input_name: str, blob: np.ndarray) -> np.ndarray:
    infer_request.infer({input_name: blob})
    # Use first output tensor for depth-like visualization.
    out_tensor = infer_request.get_output_tensor(0)
    return extract_depth_map(out_tensor.data)


def compose_display_frame(frame_bgr: np.ndarray, depth_u8: np.ndarray) -> np.ndarray:
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(
        depth_color, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    left = frame_bgr.copy()
    right = depth_color
    cv2.putText(left, "Input", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(right, "Depth", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return np.hstack([left, right])


class FrameSource:
    def __init__(
        self,
        mode: str,
        input_path: Optional[str],
        camera_id: int,
        loop_video: bool,
        synthetic_size: Tuple[int, int],
    ) -> None:
        self.mode = mode
        self.loop_video = loop_video
        self.capture: Optional[cv2.VideoCapture] = None
        self.image: Optional[np.ndarray] = None
        self.synthetic_frame = np.random.randint(
            0, 256, (synthetic_size[1], synthetic_size[0], 3), dtype=np.uint8
        )

        if mode == "image":
            if not input_path:
                raise ValueError("--input-path is required when --input-mode=image")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")
            self.image = image
        elif mode == "video":
            if not input_path:
                raise ValueError("--input-path is required when --input-mode=video")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {input_path}")
            self.capture = cap
        elif mode == "webcam":
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise ValueError(f"Failed to open webcam index: {camera_id}")
            self.capture = cap
        elif mode == "synthetic":
            pass
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

    def reset(self) -> None:
        if self.mode == "video" and self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def can_reset(self) -> bool:
        return self.mode in {"video", "image", "synthetic"}

    def read(self) -> Optional[np.ndarray]:
        if self.mode == "image":
            return self.image.copy() if self.image is not None else None
        if self.mode == "synthetic":
            return self.synthetic_frame

        assert self.capture is not None
        ok, frame = self.capture.read()
        if ok:
            return frame

        if self.mode == "video" and self.loop_video:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if ok:
                return frame
        return None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


@dataclass
class BenchResult:
    device: str
    frames: int
    total_seconds: float
    fps: float
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    compile_seconds: float
    error: Optional[str] = None


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def run_benchmark(
    core: ov.Core,
    model: ov.Model,
    input_spec: InputSpec,
    args: argparse.Namespace,
    device: str,
) -> BenchResult:
    source = FrameSource(
        mode=args.input_mode,
        input_path=args.input_path,
        camera_id=args.camera_id,
        loop_video=args.loop,
        synthetic_size=tuple(args.synthetic_size),
    )

    compile_config: Dict[str, str] = {"PERFORMANCE_HINT": args.perf_hint}
    if args.num_streams:
        compile_config["NUM_STREAMS"] = str(args.num_streams)

    try:
        compile_start = time.perf_counter()
        compiled_model = core.compile_model(model, device_name=device, config=compile_config)
        compile_seconds = time.perf_counter() - compile_start
    except Exception as exc:  # pylint: disable=broad-except
        source.release()
        return BenchResult(
            device=device,
            frames=0,
            total_seconds=0.0,
            fps=0.0,
            mean_ms=0.0,
            median_ms=0.0,
            p90_ms=0.0,
            p99_ms=0.0,
            compile_seconds=0.0,
            error=str(exc),
        )

    infer_request = compiled_model.create_infer_request()

    # Warmup to reduce one-time startup effect.
    for _ in range(max(0, args.warmup)):
        frame = source.read()
        if frame is None:
            break
        blob = preprocess_frame(frame, input_spec)
        infer_request.infer({input_spec.name: blob})

    if source.can_reset():
        source.reset()

    latencies_ms: List[float] = []
    bench_start = time.perf_counter()
    stop_requested = False

    if args.num_requests <= 1:
        frames_done = 0
        while True:
            if args.iterations > 0 and frames_done >= args.iterations:
                break
            if args.duration > 0 and (time.perf_counter() - bench_start) >= args.duration:
                break

            frame = source.read()
            if frame is None:
                break

            blob = preprocess_frame(frame, input_spec)
            t0 = time.perf_counter()
            depth_u8 = infer_depth_map(infer_request, input_spec.name, blob)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            if args.display:
                show = compose_display_frame(frame, depth_u8)
                cv2.imshow("Input + Depth", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                    break

            frames_done += 1
    else:
        queue = ov.AsyncInferQueue(compiled_model, args.num_requests)
        lock = threading.Lock()
        counters = {"done": 0}

        def callback(_request: ov.InferRequest, userdata: float) -> None:
            elapsed_ms = (time.perf_counter() - userdata) * 1000.0
            with lock:
                counters["done"] += 1
                latencies_ms.append(elapsed_ms)

        queue.set_callback(callback)

        submitted = 0
        while True:
            if args.iterations > 0 and submitted >= args.iterations:
                break
            if args.duration > 0 and (time.perf_counter() - bench_start) >= args.duration:
                break

            frame = source.read()
            if frame is None:
                break

            if args.display:
                cv2.imshow("Input", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                    break

            blob = preprocess_frame(frame, input_spec)
            queue.start_async({input_spec.name: blob}, userdata=time.perf_counter())
            submitted += 1

        queue.wait_all()

    total_seconds = time.perf_counter() - bench_start
    source.release()
    if args.display:
        cv2.destroyAllWindows()

    frames = len(latencies_ms)
    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    median_ms = statistics.median(latencies_ms) if latencies_ms else 0.0

    if stop_requested and args.verbose:
        print(f"[{device}] stopped by user via display window.")

    return BenchResult(
        device=device,
        frames=frames,
        total_seconds=total_seconds,
        fps=(frames / total_seconds) if total_seconds > 0 else 0.0,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p90_ms=percentile(latencies_ms, 90),
        p99_ms=percentile(latencies_ms, 99),
        compile_seconds=compile_seconds,
    )


def print_result(result: BenchResult) -> None:
    if result.error:
        print(f"[{result.device}] ERROR: {result.error}")
        return

    print(
        f"[{result.device}] frames={result.frames} total={result.total_seconds:.3f}s "
        f"fps={result.fps:.2f} compile={result.compile_seconds:.3f}s "
        f"lat(ms): mean={result.mean_ms:.2f} p50={result.median_ms:.2f} "
        f"p90={result.p90_ms:.2f} p99={result.p99_ms:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenVINO inference stress test for NPU/GPU/CPU with webcam/video/image input."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to OpenVINO model XML file.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["webcam", "video", "image", "synthetic"],
        default="synthetic",
        help="Input source mode.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Image/video path when input-mode is image or video.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--loop", action="store_true", help="Loop video file when it ends.")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Target device (e.g. NPU, GPU, CPU, AUTO:NPU,GPU,CPU).",
    )
    parser.add_argument(
        "--fallback-devices",
        type=str,
        default="",
        help="Try devices in order and pick first available (e.g. NPU,GPU,CPU).",
    )
    parser.add_argument(
        "--benchmark-devices",
        type=str,
        default="",
        help="Benchmark all listed devices sequentially (e.g. NPU,GPU,CPU).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of frames to run (0 means unlimited until duration/end-of-input).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Benchmark duration in seconds (0 means unlimited until iterations/end-of-input).",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before timing.")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="OpenVINO async infer requests. >1 enables async mode.",
    )
    parser.add_argument(
        "--perf-hint",
        type=str,
        default="THROUGHPUT",
        choices=["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
        help="OpenVINO PERFORMANCE_HINT.",
    )
    parser.add_argument(
        "--num-streams",
        type=str,
        default="",
        help="OpenVINO NUM_STREAMS value (optional).",
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Synthetic frame size.",
    )
    parser.add_argument(
        "--dynamic-input-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Fallback input size when model shape has dynamic H/W.",
    )
    parser.add_argument("--display", action="store_true", help="Show input frames while running.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.iterations <= 0 and args.duration <= 0:
        raise ValueError("At least one stop condition is required: --iterations > 0 or --duration > 0")

    core = ov.Core()
    model = core.read_model(str(model_path))
    input_spec = resolve_input_spec(model, tuple(args.dynamic_input_size))

    if args.display and args.num_requests > 1:
        print(
            "Display mode uses synchronous inference for stable visualization. "
            "Overriding --num-requests to 1."
        )
        args.num_requests = 1

    if args.verbose:
        print(f"Detected devices: {core.available_devices}")
        print(
            f"Model input: name={input_spec.name} shape={input_spec.shape} "
            f"layout={input_spec.layout} resize={input_spec.width}x{input_spec.height}"
        )

    bench_device_list = parse_device_list(args.benchmark_devices)
    if bench_device_list:
        devices: List[str] = []
        for item in bench_device_list:
            cand = canonical_device_name(core, item)
            if cand:
                devices.append(cand)
            else:
                devices.append(item)
    else:
        fallback = parse_device_list(args.fallback_devices)
        selected = pick_device(core, args.device, fallback)
        devices = [selected]

    print(f"Running benchmark on: {devices}")
    results: List[BenchResult] = []
    for device in devices:
        if args.verbose:
            print(f"Starting device: {device}")
        result = run_benchmark(core, model, input_spec, args, device)
        print_result(result)
        results.append(result)

    ok = [r for r in results if not r.error]
    if len(ok) > 1:
        print("\nSummary (sorted by FPS):")
        for item in sorted(ok, key=lambda x: x.fps, reverse=True):
            print(f"{item.device:12s} fps={item.fps:10.2f} mean_ms={item.mean_ms:10.2f} frames={item.frames}")


if __name__ == "__main__":
    main()
