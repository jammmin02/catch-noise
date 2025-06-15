import os
import subprocess
import argparse

# 기본 경로 설정 (CNN-only v2 기준)
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 경로 설정
ONNX_PATH = os.path.join(OUTPUT_DIR, "best_model.onnx")
ENGINE_PATH = os.path.join(OUTPUT_DIR, "best_model_fp16.trt")

# trtexec 기본 명령어 템플릿
TRTEXEC_TEMPLATE = (
    "trtexec --onnx={onnx_path} "
    "--saveEngine={engine_path} "
    "--explicitBatch "
    "--fp16 "
    "--workspace=2048 "
)

def convert_to_trt():
    # 명령어 생성
    command = TRTEXEC_TEMPLATE.format(
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH
    )

    print(f"TensorRT 변환 명령어:\n{command}\n")

    try:
        # 명령어 실행 (trtexec는 시스템에 설치되어 있어야 함)
        subprocess.run(command, shell=True, check=True)
        print(f"TensorRT 엔진 생성 완료 → {ENGINE_PATH}")

    except subprocess.CalledProcessError as e:
        print(f"TensorRT 변환 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT 변환 스크립트")
    parser.add_argument('--convert', action='store_true', help='TensorRT 변환 수행')

    args = parser.parse_args()

    if args.convert:
        convert_to_trt()
    else:
        print("변환을 실행하려면 --convert 옵션을 사용하세요.")
