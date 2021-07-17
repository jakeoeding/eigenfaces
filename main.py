import numpy as np
import matplotlib.pyplot as plt

def load_data():
    raw_data = np.genfromtxt('olivetti_faces.csv', delimiter=',')
    data_without_headers = np.delete(raw_data, 0, axis=0)
    labels = data_without_headers[: ,-1]
    face_data = np.delete(data_without_headers, -1, axis=1)
    return face_data, labels

def compute_mean_image(face_data):
    return np.average(face_data, axis=0)

def save_face(output_name, img_data):
    plt.imsave(output_name, img_data, cmap='gray')

def compute_eigenfaces(face_data, n=20):
    mean_face = compute_mean_image(face_data)
    faces_less_mean = face_data - mean_face
    U, _, _ = np.linalg.svd(faces_less_mean.T)
    return U[:, 0:n]

face_data, labels = load_data()
eigenfaces = compute_eigenfaces(face_data, 400)
mean_face = compute_mean_image(face_data)
faces_less_mean = face_data - mean_face
weights = np.matmul(faces_less_mean, eigenfaces)
