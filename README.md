# Study01_MNIST

PyTorch로 만든 손글씨 숫자(0~9) 인식 CNN입니다.

## 구성 파일
- `mnist_train.py` : MNIST 데이터셋을 내려받아 CNN을 학습하고 `mnist_cnn.pt` 로 가중치를 저장합니다.
- `mnist_predict.py` : 학습된 가중치를 불러와 손글씨 이미지를 인식합니다.
- `mnist_cnn.pt` : 학습이 끝난 모델 가중치 (테스트 정확도 약 98.8%).

## 사용법
```bash
pip install torch torchvision numpy pillow

# 학습 (이미 mnist_cnn.pt 가 있으므로 생략 가능)
python mnist_train.py

# 손글씨 이미지 인식
python mnist_predict.py 내숫자.png

# MNIST 테스트 이미지로 시연
python mnist_predict.py --샘플
```

모든 코드와 주석은 한글로 작성되어 있습니다.
