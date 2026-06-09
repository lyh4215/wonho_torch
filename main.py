import numpy as np
import matplotlib.pyplot as plt
from data import DataLoader, TensorDataset
from modules import *
from criterion import *
from optimizer import *
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
def test_BCELoss():
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

def test_softmax_CELoss():


    np.random.seed(42)

    N = 300

    X0 = np.random.randn(N, 2) * 0.3 + np.array([0.0, 1.0])
    X1 = np.random.randn(N, 2) * 0.3 + np.array([-1.0, -1.0])
    X2 = np.random.randn(N, 2) * 0.3 + np.array([1.0, -1.0])

    X = np.vstack([X0, X1, X2]).astype(float)
    Y = np.array([0] * N + [1] * N + [2] * N)

    print(X.shape)  # (900, 2)
    print(Y.shape)  # (900,)
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 3)
    )

    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)

    batch_size = 32

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = 1000

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion.forward(logits, Y_batch)

            dy = criterion.backward()
            model.backward(dy)

            optimizer.step()

            total_loss += loss

        if epoch % 100 == 0:
            avg_loss = total_loss / len(dataloader)

            logits_all = model(X)
            probs_all = softmax(logits_all)
            pred_labels = np.argmax(probs_all, axis=1)

            acc = np.mean(pred_labels == Y)

            print(f"epoch {epoch:04d} | loss {avg_loss:.6f} | acc {acc:.4f}")

def test_mnist_softmax_CELoss():
    from mnist_loader import load_mnist_numpy

    np.random.seed(42)

    # 1. MNIST load
    X_train, Y_train, X_test, Y_test = load_mnist_numpy()

    # 2. 전처리
    # X_train: (60000, 28, 28) -> (60000, 784)
    X_train = X_train.reshape(-1, 784).astype(float) / 255.0
    X_test = X_test.reshape(-1, 784).astype(float) / 255.0

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-7

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    print(X_train.shape)  # (60000, 784)
    print(Y_train.shape)  # (60000,)
    print(X_test.shape)   # (10000, 784)
    print(Y_test.shape)   # (10000,)

    # 처음엔 전체 MNIST가 느릴 수 있으니 일부만 사용
    train_limit = 60000
    test_limit = 10000

    train_idx = np.random.permutation(len(X_train))[:train_limit]
    test_idx = np.random.permutation(len(X_test))[:test_limit]

    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]
    print(np.bincount(Y_train[:12000]))
    # 3. Model
    # 입력: 784차원
    # 출력: 숫자 0~9, 총 10개 class
    model = Sequential(
        Linear(784, 256),
        BatchNorm1D(256),
        ReLU(),
        Dropout(0.2),

        Linear(256, 128),
        BatchNorm1D(128),
        ReLU(),
        Dropout(0.2),

        Linear(128, 10)
    )

    criterion = SoftmaxCrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    #optimizer = SGD(model.parameters(), lr=1e-1)

    batch_size = 64

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = 20
    

    for epoch in range(epochs):
        total_loss = 0
        total_count = 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for X_batch, Y_batch in pbar:
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion.forward(logits, Y_batch)

            dy = criterion.backward()
            model.backward(dy)

            optimizer.step()

            total_loss += loss * len(X_batch)
            total_count += len(X_batch)
            pbar.set_postfix(loss=f"{loss:.6f}")

        avg_loss = total_loss / total_count

        # train accuracy
        train_logits = model(X_train)
        train_probs = softmax(train_logits)
        train_pred = np.argmax(train_probs, axis=1)
        train_acc = np.mean(train_pred == Y_train)

        # test accuracy
        test_logits = model(X_test)
        test_probs = softmax(test_logits)
        test_pred = np.argmax(test_probs, axis=1)
        test_acc = np.mean(test_pred == Y_test)

        print(
            f"epoch {epoch:03d} | "
            f"loss {avg_loss:.6f} | "
            f"train_acc {train_acc:.4f} | "
            f"test_acc {test_acc:.4f}"
        )
        model.eval()

        train_logits = model(X_train)
        test_logits = model(X_test)

def test_mnist_cnn():
    from mnist_loader import load_mnist_numpy

    np.random.seed(42)

    # 1. MNIST load
    X_train, Y_train, X_test, Y_test = load_mnist_numpy()

    # 2. 전처리
    # CNN input: (N, C, H, W)
    # MNIST: (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28).astype(float) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28).astype(float) / 255.0

    # 입력 정규화
    # channel/image 구조를 유지한 채로 pixel 위치별 mean/std 계산
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-7

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    print(X_train.shape)  # (60000, 1, 28, 28)
    print(Y_train.shape)  # (60000,)
    print(X_test.shape)   # (10000, 1, 28, 28)
    print(Y_test.shape)   # (10000,)

    train_limit = 60000
    test_limit = 10000

    train_idx = np.random.permutation(len(X_train))[:train_limit]
    test_idx = np.random.permutation(len(X_test))[:test_limit]

    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]
    print(np.bincount(Y_train[:12000]))
    # 3. Model
    # 입력: 784차원
    # 출력: 숫자 0~9, 총 10개 class
    model = SimpleCNN()
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    batch_size = 64

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epochs = 20
    def evaluate(model, X, Y, batch_size=256):
        model.eval()

        correct = 0
        total = 0

        for start in range(0, len(X), batch_size):
            end = start + batch_size

            X_batch = X[start:end]
            Y_batch = Y[start:end]

            logits = model(X_batch)
            pred = np.argmax(logits, axis=1)

            correct += np.sum(pred == Y_batch)
            total += len(Y_batch)

        return correct / total

    for epoch in range(epochs):
        total_loss = 0
        total_count = 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for X_batch, Y_batch in pbar:
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion.forward(logits, Y_batch)

            dy = criterion.backward()
            model.backward(dy)

            optimizer.step()

            total_loss += loss * len(X_batch)
            total_count += len(X_batch)
            pbar.set_postfix(loss=f"{loss:.6f}")

        avg_loss = total_loss / total_count

        train_acc = evaluate(model, X_train, Y_train, batch_size=256)
        test_acc = evaluate(model, X_test, Y_test, batch_size=256)

        print(
            f"epoch {epoch:03d} | "
            f"loss {avg_loss:.6f} | "
            f"train_acc {train_acc:.4f} | "
            f"test_acc {test_acc:.4f}"
        )

if __name__ == "__main__":
    test_mnist_cnn()