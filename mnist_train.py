"""
손글씨 숫자(MNIST) 인식 CNN 학습 스크립트 (PyTorch)

실행 방법:
    python mnist_train.py

동작:
    1. MNIST 데이터셋을 data/ 폴더에 내려받는다 (이미 있으면 재사용).
    2. CNN 모델을 학습한다.
    3. 테스트 정확도를 확인하고 가중치를 mnist_cnn.pt 로 저장한다.
"""

import gzip
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
데이터_폴더 = "data"
가중치_파일 = "mnist_cnn.pt"
에폭_수 = 3
배치_크기 = 128
학습률 = 1e-3
난수_시드 = 42

# 공식 MNIST 파일 미러 (GitHub)
다운로드_주소 = "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data"
데이터_파일 = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


# ----------------------------------------------------------------------
# 모델 정의 (predict 스크립트에서도 그대로 가져다 쓴다)
# ----------------------------------------------------------------------
class MnistCNN(nn.Module):
    """합성곱 2층 + 완전연결 2층으로 구성된 간단한 CNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14x14 -> 7x7
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)  # 각 숫자(0~9)에 대한 로짓


# ----------------------------------------------------------------------
# 데이터 준비
# ----------------------------------------------------------------------
def 데이터_내려받기():
    """필요한 MNIST 파일이 없으면 내려받는다."""
    os.makedirs(데이터_폴더, exist_ok=True)
    for 이름 in 데이터_파일:
        경로 = os.path.join(데이터_폴더, 이름)
        if not os.path.exists(경로):
            print(f"내려받는 중: {이름}")
            urllib.request.urlretrieve(f"{다운로드_주소}/{이름}", 경로)


def 이미지_읽기(경로):
    """IDX 형식의 이미지 파일을 (N, 1, 28, 28) float32 텐서(0~1)로 읽는다."""
    with gzip.open(경로, "rb") as f:
        _, 개수, 행, 열 = struct.unpack(">IIII", f.read(16))
        데이터 = np.frombuffer(f.read(), dtype=np.uint8).reshape(개수, 1, 행, 열)
    return torch.from_numpy(데이터.copy()).float() / 255.0


def 라벨_읽기(경로):
    """IDX 형식의 라벨 파일을 (N,) int64 텐서로 읽는다."""
    with gzip.open(경로, "rb") as f:
        _, 개수 = struct.unpack(">II", f.read(8))
        데이터 = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(데이터.copy()).long()


def 데이터_불러오기():
    데이터_내려받기()
    경로 = lambda 이름: os.path.join(데이터_폴더, 이름)
    학습_x = 이미지_읽기(경로(데이터_파일[0]))
    학습_y = 라벨_읽기(경로(데이터_파일[1]))
    시험_x = 이미지_읽기(경로(데이터_파일[2]))
    시험_y = 라벨_읽기(경로(데이터_파일[3]))
    return 학습_x, 학습_y, 시험_x, 시험_y


# 정규화에 쓰는 MNIST 평균/표준편차 (predict 스크립트와 동일해야 한다)
평균, 표준편차 = 0.1307, 0.3081


# ----------------------------------------------------------------------
# 학습 / 평가
# ----------------------------------------------------------------------
def 평가(모델, 로더, 장치):
    """테스트 정확도(%)를 계산한다."""
    모델.eval()
    맞은_수 = 0
    전체_수 = 0
    with torch.no_grad():
        for x, y in 로더:
            x, y = x.to(장치), y.to(장치)
            예측 = 모델(x).argmax(dim=1)
            맞은_수 += (예측 == y).sum().item()
            전체_수 += y.size(0)
    return 100.0 * 맞은_수 / 전체_수


def main():
    torch.manual_seed(난수_시드)
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {장치}")

    학습_x, 학습_y, 시험_x, 시험_y = 데이터_불러오기()
    print(f"학습 데이터: {len(학습_x)}장, 테스트 데이터: {len(시험_x)}장")

    # 정규화 적용
    학습_x = (학습_x - 평균) / 표준편차
    시험_x = (시험_x - 평균) / 표준편차

    학습_로더 = DataLoader(TensorDataset(학습_x, 학습_y), batch_size=배치_크기, shuffle=True)
    시험_로더 = DataLoader(TensorDataset(시험_x, 시험_y), batch_size=1000)

    모델 = MnistCNN().to(장치)
    최적화 = torch.optim.Adam(모델.parameters(), lr=학습률)

    for 에폭 in range(1, 에폭_수 + 1):
        모델.train()
        시작 = time.time()
        손실_합 = 0.0
        for x, y in 학습_로더:
            x, y = x.to(장치), y.to(장치)
            최적화.zero_grad()
            손실 = F.cross_entropy(모델(x), y)
            손실.backward()
            최적화.step()
            손실_합 += 손실.item() * x.size(0)

        평균_손실 = 손실_합 / len(학습_x)
        정확도 = 평가(모델, 시험_로더, 장치)
        print(f"[에폭 {에폭}/{에폭_수}] 손실 {평균_손실:.4f} | "
              f"테스트 정확도 {정확도:.2f}% | {time.time() - 시작:.1f}초")

    # 학습된 가중치 저장
    torch.save(모델.state_dict(), 가중치_파일)
    print(f"가중치를 '{가중치_파일}' 로 저장했습니다.")


if __name__ == "__main__":
    main()
