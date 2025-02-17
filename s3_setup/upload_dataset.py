import os

import boto3

# Inicializamos el cliente S3
s3 = boto3.client("s3")

# Definimos los parámetros
bucket_name = "deepsampler "
folder_path = "dataset"  # Ruta de la carpeta local

# Iteramos a través de todos los archivos en la carpeta
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Construimos la ruta local y la ruta en el S3
        local_file_path = os.path.join(root, file)
        s3_key = os.path.relpath(local_file_path, folder_path)  # Ruta relativa en S3

        # Subimos el archivo al S3
        s3.upload_file(local_file_path, bucket_name, f"datasets/{s3_key}")

        # Imprimimos el path de S3 del archivo subido
        print(f"Archivo subido a: s3://{bucket_name}/datasets/{s3_key}")
