"""
손글씨 숫자 이미지 인식 스크립트 (PyTorch)

먼저 mnist_train.py 를 실행해 mnist_cnn.pt 를 만들어 두어야 한다.

사용 방법:
    python mnist_predict.py 내숫자.png            # 이미지 한 장 인식
    python mnist_predict.py a.png b.png c.png     # 여러 장 인식
    python mnist_predict.py --샘플                # MNIST 테스트 이미지로 시연

이미지 조건:
    - 숫자 하나가 들어 있는 이미지 (png, jpg 등)
    - 흰 종이에 검은 펜이든, 검은 배경에 흰 글씨든 상관없다 (자동 판별).
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from mnist_train import MnistCNN, 가중치_파일, 평균, 표준편차


def 이미지_전처리(경로):
    """
    사용자가 쓴 손글씨 이미지를 MNIST 형식(28x28, 검은 배경에 흰 글씨)으로 바꾼다.

    1. 흑백으로 변환
    2. 배경이 밝으면 색을 반전 (MNIST는 검은 배경 + 흰 글씨)
    3. 글씨가 있는 영역만 잘라내기
    4. 가로세로 비율을 유지한 채 20x20 안에 맞추기
    5. 무게중심이 28x28 정중앙에 오도록 배치
    """
    그림 = np.array(Image.open(경로).convert("L"), dtype=np.float32)

    # 배경이 밝은 이미지(흰 종이)라면 반전한다
    if 그림.mean() > 127:
        그림 = 255.0 - 그림

    # 아주 옅은 잡음은 제거하고, 글씨가 있는 영역의 경계 상자를 구한다
    그림[그림 < 30] = 0
    행, 열 = np.where(그림 > 0)
    if len(행) == 0:
        raise ValueError(f"'{경로}' 에서 글씨를 찾지 못했습니다.")
    그림 = 그림[행.min():행.max() + 1, 열.min():열.max() + 1]

    # 긴 변이 20픽셀이 되도록 비율 유지 축소
    높이, 너비 = 그림.shape
    비율 = 20.0 / max(높이, 너비)
    새_높이, 새_너비 = max(1, round(높이 * 비율)), max(1, round(너비 * 비율))
    그림 = np.array(
        Image.fromarray(그림.astype(np.uint8)).resize((새_너비, 새_높이), Image.LANCZOS),
        dtype=np.float32,
    )

    # 28x28 캔버스 중앙에 우선 배치
    캔버스 = np.zeros((28, 28), dtype=np.float32)
    위 = (28 - 새_높이) // 2
    왼쪽 = (28 - 새_너비) // 2
    캔버스[위:위 + 새_높이, 왼쪽:왼쪽 + 새_너비] = 그림

    # 무게중심을 (14, 14)로 옮긴다 (MNIST와 같은 방식)
    ys, xs = np.indices(캔버스.shape)
    총합 = 캔버스.sum()
    중심_y = (ys * 캔버스).sum() / 총합
    중심_x = (xs * 캔버스).sum() / 총합
    이동_y, 이동_x = round(14 - 중심_y), round(14 - 중심_x)
    캔버스 = np.roll(캔버스, (이동_y, 이동_x), axis=(0, 1))

    return 캔버스 / 255.0  # 0~1 범위


def 모델_불러오기():
    if not os.path.exists(가중치_파일):
        raise FileNotFoundError(
            f"'{가중치_파일}' 이 없습니다. 먼저 'python mnist_train.py' 를 실행하세요."
        )
    모델 = MnistCNN()
    모델.load_state_dict(torch.load(가중치_파일, map_location="cpu"))
    모델.eval()  # 추론 모드 (드롭아웃 끄기)
    return 모델


def 숫자_인식(모델, 배열_28x28):
    """28x28(0~1) 배열을 받아 (예측 숫자, 확신도, 전체 확률)을 돌려준다."""
    입력 = torch.from_numpy(배열_28x28).float().reshape(1, 1, 28, 28)
    입력 = (입력 - 평균) / 표준편차
    with torch.no_grad():
        확률 = torch.softmax(모델(입력), dim=1)[0]
    숫자 = int(확률.argmax())
    return 숫자, float(확률[숫자]), 확률


def 샘플_시연(모델):
    """MNIST 테스트 이미지를 흰 종이에 쓴 손글씨처럼 저장한 뒤 인식해 본다."""
    from mnist_train import 데이터_불러오기

    _, _, 시험_x, 시험_y = 데이터_불러오기()
    os.makedirs("샘플", exist_ok=True)
    맞은_수 = 0
    개수 = 10
    for i in range(개수):
        # 검은 글씨 + 흰 배경, 280x280 크기로 저장 (실제 손글씨 사진과 비슷한 형태)
        배열 = (255 - 시험_x[i, 0].numpy() * 255).astype(np.uint8)
        경로 = f"샘플/샘플_{i}.png"
        Image.fromarray(배열).resize((280, 280), Image.BICUBIC).save(경로)

        숫자, 확신도, _ = 숫자_인식(모델, 이미지_전처리(경로))
        정답 = int(시험_y[i])
        맞은_수 += int(숫자 == 정답)
        표시 = "O" if 숫자 == 정답 else "X"
        print(f"{경로}: 정답 {정답} / 예측 {숫자} (확신도 {확신도 * 100:.1f}%) {표시}")
    print(f"\n샘플 {개수}장 중 {맞은_수}장 정답")


def main():
    파서 = argparse.ArgumentParser(description="손글씨 숫자 인식")
    파서.add_argument("이미지", nargs="*", help="인식할 이미지 파일 경로")
    파서.add_argument("--샘플", action="store_true", help="MNIST 테스트 이미지로 시연")
    인자 = 파서.parse_args()

    모델 = 모델_불러오기()

    if 인자.샘플:
        샘플_시연(모델)
        return
    if not 인자.이미지:
        파서.print_help()
        return

    for 경로 in 인자.이미지:
        숫자, 확신도, 확률 = 숫자_인식(모델, 이미지_전처리(경로))
        상위 = torch.topk(확률, 3)
        후보 = ", ".join(f"{int(i)}({float(p) * 100:.1f}%)" for p, i in zip(상위.values, 상위.indices))
        print(f"{경로}: 예측 숫자 = {숫자} (확신도 {확신도 * 100:.1f}%)  |  상위 후보: {후보}")


if __name__ == "__main__":
    main()
