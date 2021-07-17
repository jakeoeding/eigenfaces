import numpy as np

class FaceRecognizer:
    def __init__(self):
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

    def find_eigenfaces(self, n=80):
        self.faces_less_mean = self.face_data - self.mean_face
        U, _, _ = np.linalg.svd(self.faces_less_mean.T)
        self.eigenfaces = U[:, 0:n]


if __name__ == '__main__':
    fr = FaceRecognizer()
    fr.train()
