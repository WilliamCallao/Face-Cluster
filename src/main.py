import os
from src.face_extractor import extract_faces
from src.face_cluster import cluster_faces

def main():
    default_input_path = "D:\\Archivos\\Cluster\\Input"
    default_output_path = "D:\\Archivos\\Cluster\\Output"

    input_path = input(f"Ingrese la ruta de la carpeta con las imágenes de entrada (predeterminado: {default_input_path}): ") or default_input_path
    output_path = input(f"Ingrese la ruta de la carpeta para guardar los resultados (predeterminado: {default_output_path}): ") or default_output_path

    if not os.path.exists(input_path):
        print("La ruta de entrada no existe.")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Extrayendo rostros de las imágenes...")
    faces_with_originals = extract_faces(input_path)

    if not faces_with_originals:
        print("No se detectaron rostros en las imágenes de entrada.")
        return

    faces, original_images = zip(*faces_with_originals)

    print("Agrupando rostros similares...")
    cluster_faces(faces, output_path, original_images)

    print("Proceso completado. Los rostros y las imágenes originales se han guardado en la carpeta de salida.")

if __name__ == "__main__":
    main()
