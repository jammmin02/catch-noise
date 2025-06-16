# scripts/build_engine.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

onnx_file = "/workspace/data/only-cnn.onnx"
engine_file = "/workspace/data/only-cnn.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 30  # 1GB
        builder.fp16_mode = True

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine = builder.build_cuda_engine(network)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"TensorRT 엔진 저장 완료: {engine_path}")

if __name__ == "__main__":
    if os.path.exists(onnx_file):
        build_engine(onnx_file, engine_file)
    else:
        print("ONNX 파일이 존재하지 않습니다:", onnx_file)
