import numpy as np
import matplotlib.pyplot as plt
from data import DataLoader, TensorDataset
from modules import *
from criterion import *
from optimizer import SGD
from tqdm import tqdm


def test_linear():
    batch_size = 10

    np.random.seed(42)

    # 데이터 개수
    N = 200

    # 입력 X: -3 ~ 3 사이 값
    X = np.linspace(-3, 3, N).reshape(-1, 1)

    # 정답 Y: 비선형 함수
    Y = np.sin(X) + 0.3 * X**2

    # 약간의 노이즈 추가
    Y = Y + 0.1 * np.random.randn(N, 1)

    print(X.shape)  # (200, 1)
    print(Y.shape)  # (200, 1)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = Sequential(
        Linear(1, 32),
        ReLU(),
        Linear(32, 32),
        ReLU(),
        Linear(32, 1)
    )
    criterion = MSELoss()
    optimizer = SGD(model.parameters())
    print(model)

    #train
    epochs = 1000

    for epoch in range(epochs):
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for X_batch, Y_batch in pbar:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion.forward(y_pred, Y_batch)

            dy = criterion.backward()
            model.backward(dy)

            optimizer.step()

            total_loss += loss

            # 현재 batch loss 표시
            pbar.set_postfix(loss=f"{loss:.6f}")

        avg_loss = total_loss / len(dataloader)

        # epoch 끝난 뒤 평균 loss 표시
        if epoch % 100 == 0:
            print(f"epoch: {epoch:4d} | avg_loss: {avg_loss:.6f}")

    # 저장
    state = model.state_dict()

    for key, value in state.items():
        print(key, value.shape)

    state = model.state_dict()
    np.savez("model_state.npz", **state)

    loaded = np.load("model_state.npz")

    state = {}
    for key in loaded.files:
        state[key] = loaded[key]

    model.load_state_dict(state)
    # 전체 X에 대해 예측
    Y_pred = model(X)

    # 시각화를 위해 X 기준 정렬
    sorted_idx = np.argsort(X[:, 0])

    X_sorted = X[sorted_idx]
    Y_sorted = Y[sorted_idx]
    Y_pred_sorted = Y_pred[sorted_idx]

    plt.figure(figsize=(8, 5))

    # 실제 데이터
    plt.scatter(X_sorted, Y_sorted, label="True data", alpha=0.5)

    # 모델 예측
    plt.plot(X_sorted, Y_pred_sorted, label="Model prediction", linewidth=2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("True data vs Model prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig("result.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    np.random.seed(42)

    N = 500

    X = np.random.uniform(-2, 2, size=(N, 2))

    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

    Y = (r < 1.0).astype(float).reshape(-1, 1)

    print(X.shape)  # (500, 2)
    print(Y.shape)  # (500, 1)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 1),
        Sigmoid()
    )

    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=1e-2)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 1000

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion.forward(y_pred, Y_batch)

            dy = criterion.backward()
            model.backward(dy)

            optimizer.step()

            total_loss += loss

        if epoch % 100 == 0:
            avg_loss = total_loss / len(dataloader)

            y_pred_all = model(X)
            pred_label = (y_pred_all > 0.5).astype(float)
            acc = np.mean(pred_label == Y)

            print(f"epoch {epoch:04d} | loss {avg_loss:.6f} | acc {acc:.4f}")

        
    y_pred = model(X)

    print(y_pred[:10])
    print((y_pred[:10] > 0.5).astype(int))
    print(Y[:10].astype(int))