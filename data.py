import numpy as np    
class TensorDataset:
    def __init__(self, X: np.ndarray, Y : np.ndarray):
        assert len(X) == len(Y), "X and Y must have same length"

        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
class DataLoader:
    def __init__ (self, dataset: TensorDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            indices = np.random.permutation(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start: start+self.batch_size]

            X_batch = []
            Y_batch = []

            for idx in batch_indices:
                x, y = self.dataset[idx]
                X_batch.append(x)
                Y_batch.append(y)
            yield np.array(X_batch), np.array(Y_batch)
        
    def __len__(self):
        return int(np.ceil(len(self.dataset)/ self.batch_size))