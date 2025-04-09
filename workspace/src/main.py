#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CUDA 및 GPU 작동 확인 스크립트
"""

import os
import sys
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import platform
import numpy as np
from datetime import datetime


def print_system_info():
    """
    시스템 정보 및 CUDA 가용성을 출력합니다.
    """
    print("\n" + "="*50)
    print("시스템 및 CUDA 정보")
    print("="*50)
    
    print(f"현재 날짜 및 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"운영체제: {platform.platform()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '사용 불가'}")
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  메모리 할당: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  메모리 예약: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("CUDA를 사용할 수 없습니다. CPU만 사용 가능합니다.")
    
    print("="*50 + "\n")


def test_tensor_operations():
    """
    기본 텐서 연산 테스트를 수행합니다.
    """
    print("\n" + "="*50)
    print("기본 텐서 연산 테스트")
    print("="*50)
    
    # CPU 텐서 생성
    a_cpu = torch.randn(1000, 1000)
    b_cpu = torch.randn(1000, 1000)
    
    # CPU에서 행렬 곱셈
    start = datetime.now()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = (datetime.now() - start).total_seconds()
    
    print(f"CPU 행렬 곱셈 시간: {cpu_time:.6f}초")
    
    # CUDA 사용 가능한 경우 GPU에서도 테스트
    if torch.cuda.is_available():
        # GPU 텐서 생성
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # 첫 실행은 CUDA 초기화 시간을 포함하므로 무시
        torch.cuda.synchronize()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # GPU에서 행렬 곱셈
        torch.cuda.synchronize()
        start = datetime.now()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = (datetime.now() - start).total_seconds()
        
        print(f"GPU 행렬 곱셈 시간: {gpu_time:.6f}초")
        print(f"속도 향상: {cpu_time / gpu_time:.2f}x")
        
        # 결과 정확성 검증
        c_gpu_cpu = c_gpu.cpu()
        diff = torch.abs(c_cpu - c_gpu_cpu).max().item()
        print(f"CPU와 GPU 결과 최대 차이: {diff}")
    
    print("="*50 + "\n")


def test_mixed_precision():
    """
    혼합 정밀도(Mixed Precision) 테스트를 수행합니다.
    """
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 혼합 정밀도 테스트를 건너뜁니다.")
        return
    
    print("\n" + "="*50)
    print("혼합 정밀도(Mixed Precision) 테스트")
    print("="*50)
    
    # 간단한 모델 정의
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000)
    ).cuda()
    
    # 입력 데이터 생성
    x = torch.randn(100, 1000).cuda()
    target = torch.randn(100, 1000).cuda()
    
    # 기본 FP32 연산 시간 측정
    start = datetime.now()
    
    for _ in range(10):
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
    
    torch.cuda.synchronize()
    fp32_time = (datetime.now() - start).total_seconds()
    
    # 모델 재설정
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000)
    ).cuda()
    
    # 혼합 정밀도(FP16) 연산 시간 측정
    scaler = amp.GradScaler()
    start = datetime.now()
    
    for _ in range(10):
        with amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters()))
        scaler.update()
    
    torch.cuda.synchronize()
    fp16_time = (datetime.now() - start).total_seconds()
    
    print(f"FP32 연산 시간: {fp32_time:.6f}초")
    print(f"FP16 혼합 정밀도 연산 시간: {fp16_time:.6f}초")
    print(f"속도 향상: {fp32_time / fp16_time:.2f}x")
    print("="*50 + "\n")


def test_tensorrt():
    """
    TensorRT 설치 확인을 수행합니다.
    """
    print("\n" + "="*50)
    print("TensorRT 가용성 테스트")
    print("="*50)
    
    try:
        import tensorrt as trt
        print(f"TensorRT 버전: {trt.__version__}")
        print("TensorRT가 성공적으로 설치되었습니다!")
        
        # TensorRT 관련 정보 출력
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        print(f"가용 TensorRT 기능:")
        print(f"  - FP16 지원: {builder.platform_has_fast_fp16}")
        print(f"  - INT8 지원: {builder.platform_has_fast_int8}")
        
    except ImportError:
        print("TensorRT가 설치되지 않았거나 Python 환경에서 접근할 수 없습니다.")

    # torch-tensorrt 확인 시도(선택 사항)
    try:
        import torch_tensorrt
        print(f"torch-tensorrt 버전: {torch_tensorrt.__version__}")
        print("torch-tensorrt가 설치되었습니다!")
    except ImportError:
        print("torch-tensorrt가 설치되어 있지 않습니다.")
    
    print("="*50 + "\n")


def main():
    """
    메인 함수: 모든 테스트를 순차적으로 실행합니다.
    """
    print("\n" + "*"*70)
    print("*" + " "*24 + "GPU 테스트 도구" + " "*24 + "*")
    print("*"*70 + "\n")
    
    print_system_info()
    test_tensor_operations()
    
    if torch.cuda.is_available():
        test_mixed_precision()
    
    test_tensorrt()
    
    print("\n모든 테스트가 완료되었습니다.")


if __name__ == "__main__":
    main()