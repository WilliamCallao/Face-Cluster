import numpy as np
from sklearn.cluster import KMeans
import os
import cv2

def cluster_faces(faces, output_path, original_images):
    embeddings = [cv2.resize(face, (100, 100)).flatten() for face in faces]
    
    if not embeddings:
        print("No se encontraron datos para agrupar.")
        return

    embeddings = np.array(embeddings)

    num_clusters = 5 
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(embeddings)

    for i in range(num_clusters):
        cluster_folder = os.path.join(output_path, f"cluster_{i}")
        os.makedirs(cluster_folder, exist_ok=True)

        for j, (face, original_image) in enumerate(zip(faces, original_images)):
            if labels[j] == i:
                face_path = os.path.join(cluster_folder, f"face_{j}.jpg")
                cv2.imwrite(face_path, face)

                original_image_path = os.path.join(cluster_folder, f"original_image_{j}.jpg")
                cv2.imwrite(original_image_path, original_image)
