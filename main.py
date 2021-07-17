import numpy as np

class FaceRecognizer:
    def __init__(self, retention):
        self.retention = retention
        self.face_data = None
        self.labels = None
        self.mean_face = None
        self.faces_less_mean = None
        self.eigenfaces = None
        self.weights = None

    def train(self):
        self.load_data()
        self.mean_face = np.average(self.face_data, axis=0)
        self.find_eigenfaces()
        self.weights = np.matmul(self.faces_less_mean, self.eigenfaces)

    def load_data(self):
        raw_data = np.genfromtxt('olivetti_faces.csv', delimiter=',')
        data_without_headers = np.delete(raw_data, 0, axis=0)
        self.labels = data_without_headers[: ,-1]
        self.face_data = np.delete(data_without_headers, -1, axis=1)

    def find_eigenfaces(self):
        self.faces_less_mean = self.face_data - self.mean_face
        U, S, _ = np.linalg.svd(self.faces_less_mean.T)
        total_energy = S.sum()
        retention_candidates = np.where(S.cumsum() / total_energy > self.retention)
        dimensions_to_retain = retention_candidates[0][0]
        self.eigenfaces = U[:, 0:dimensions_to_retain]


if __name__ == '__main__':
    fr = FaceRecognizer(0.9)
    fr.train()
