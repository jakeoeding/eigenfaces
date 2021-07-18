import matplotlib.pyplot as plt
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
        raw_data = np.genfromtxt('olivetti_faces_augmented.csv', delimiter=',')
        data_without_headers = np.delete(raw_data, 0, axis=0)
        self.labels = data_without_headers[:, -1]
        self.face_data = np.delete(data_without_headers, -1, axis=1)

    def find_eigenfaces(self):
        self.faces_less_mean = self.face_data - self.mean_face
        U, S, _ = np.linalg.svd(self.faces_less_mean.T)
        total_energy = S.sum()
        retention_candidates = np.where(S.cumsum() / total_energy > self.retention)
        dimensions_to_retain = retention_candidates[0][0]
        self.eigenfaces = U[:, 0:dimensions_to_retain]

    def reconstruct(self, weights):
        return np.matmul(weights, self.eigenfaces.T) + self.mean_face

    def find_best_match(self, face):
        face_less_mean = face - self.mean_face
        face_weights = np.matmul(face_less_mean, self.eigenfaces)
        diffs = self.weights - face_weights
        squared_diffs = diffs ** 2
        match_index = np.argmin(squared_diffs.sum(axis=1))
        return self.reconstruct(self.weights[match_index]), self.labels[match_index]


if __name__ == '__main__':
    import utils

    fr = FaceRecognizer(0.9)
    fr.train()

    search_vector = utils.image_to_vector('test_input.jpg')
    match, match_label = fr.find_best_match(search_vector)
    plt.imsave('match.jpg', match.reshape(64, 64), cmap='gray')
