import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 자동 초기화
import os

# ✅ 경로 설정
onnx_path = "dev/jungmin/2class_noisy_vs_nonnoisy/pyTorch_v2/outputs/cnn_lstm/best_model.onnx"
trt_path = "dev/jungmin/jetson/assets/best_model.trt"

# ✅ TensorRT 빌더 초기화
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# ✅ ONNX 파일 로드 및 파싱
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        print("❌ ONNX 파싱 실패!")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)
print("✅ ONNX 파싱 완료")

# ✅ 빌더 설정
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28  # 256MB
config.set_flag(trt.BuilderFlag.FP16)  # Jetson은 FP16 최적화 가능

# ✅ 엔진 생성
engine = builder.build_engine(network, config)
assert engine is not None
print("✅ TensorRT 엔진 생성 완료")

# ✅ .trt 파일로 저장
with open(trt_path, "wb") as f:
    f.write(engine.serialize())
print(f"✅ TensorRT 엔진 저장 완료: {trt_path}")
