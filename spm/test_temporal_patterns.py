import numpy as np

from spm.temporal_patterns import discretize_dataset


DATA_PATH = r"C:\Users\Nad Jr\Downloads\wlasl25_landmarks.npz"


def main():
    data = np.load(DATA_PATH)

    X_train = data["X_train_aug"]

    print("Original shape:")
    print(X_train.shape)

    X_discretized = discretize_dataset(X_train)

    print("\nDiscretized shape:")
    print(X_discretized.shape)

    print("\nData type:")
    print(X_discretized.dtype)

    print("\nUnique states:")
    print(np.unique(X_discretized))

    print("\nFirst training sample, first 10 time steps, first 5 features:")
    print(X_discretized[0, :10, :5])


if __name__ == "__main__":
    main()