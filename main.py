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

face_data, labels = load_data()
save_face('first_face.jpg', face_data[0].reshape(64, 64))
mean_face = compute_mean_image(face_data)
save_face('mean_face.jpg', mean_face.reshape(64, 64))
