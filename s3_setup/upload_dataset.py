import os

import boto3
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Recuperar las variables definidas en el archivo .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
bucket_name = os.getenv("BUCKET_NAME")
folder_path = os.getenv("FOLDER_PATH")

# Crear una sesión de boto3 con las credenciales proporcionadas
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# Inicializar el cliente S3 usando la sesión creada
s3 = session.client("s3")

# Recorrer la carpeta local y subir todos los archivos a S3
for root, dirs, files in os.walk(folder_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        s3_key = os.path.relpath(local_file_path, folder_path)
        # Subir el archivo a S3 en la carpeta "dataset"
        s3.upload_file(local_file_path, bucket_name, f"dataset/{s3_key}")
        print(f"Archivo subido a: s3://{bucket_name}/dataset/{s3_key}")
