import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context 초기화
import numpy as np
import os

# 경로 설정
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1"
ENGINE_PATH = os.path.join(BASE_DIR, "outputs", "best_model_fp16.trt")

# 입력 정보 (전처리 기준 고정)
BATCH_SIZE = 1
CHANNELS = 1
HEIGHT = 86
WIDTH = 14
INPUT_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

def load_engine(engine_path):
    """TensorRT 엔진 로드"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine):
    """입출력 버퍼 할당 (호스트/디바이스 메모리)"""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, stream

def infer(engine, inputs, outputs, bindings, stream):
    """실제 추론 수행"""
    with engine.create_execution_context() as context:
        # 더미 입력 생성 (테스트용)
        dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32).ravel()
        np.copyto(inputs[0]['host'], dummy_input)

        # 호스트 → 디바이스 복사
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # 추론 실행
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # 디바이스 → 호스트 복사
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        # 결과 반환
        output = outputs[0]['host']
        return output

if __name__ == "__main__":
    print(f"TensorRT 엔진 로드: {ENGINE_PATH}")
    engine = load_engine(ENGINE_PATH)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    print("TensorRT 엔진 추론 테스트 시작...")
    output = infer(engine, inputs, outputs, bindings, stream)
    print("추론 결과 (logit):", output)

    # Sigmoid 적용 (확률로 변환)
    prob = 1 / (1 + np.exp(-output))
    print("추론 결과 (확률):", prob)
