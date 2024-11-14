# Backend de Tesis - APLICACIÓN DE TÉCNICAS DE INTELIGENCIA ARTIFICIAL SOBRE LOS COMENTARIOS DE LAS EVALUACIONES DOCENTES DE LA UNIVERSIDAD DEL VALLE PARA HACER TEACHING ANALYTICS

Este repositorio contiene el backend de la tesis, desarrollado con Django y Django Rest Framework (DRF). Este backend administra los datos de evaluaciones de docentes y proporciona una API REST
para interactuar con el frontend. La base de datos utilizada es PostgreSQL.

## Características
- API REST para gestionar evaluaciones, comentarios, usuarios y métricas.
- Integración con PostgreSQL para almacenamiento de datos.
- Algoritmos de análisis de sentimiento y clasificación de comentarios usando modelos de Hugging Face.
- API de Gemini para realizar extraccion de temas en los comentarios.
- Procesamiento de archivos Excel para importar datos de evaluación.

## Requisitos

- Python 3.x
- PostgreSQL

## Configurar la base de datos
1. Recordar primero que nada, crear la base de datos de PostgreSQL, se recomienda el uso de pgAdmin 4 para esto.
   
2. Dirigirse al archivo "settings.py" del proyecto para cambiar el nombre y la contraseña de la base de datos en caso de ser necesario.

   ![image](https://github.com/user-attachments/assets/ceaed080-b4f8-48cf-99f0-8d4b7db289c1)

## Instalación y Configuración

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/santa51107HD/edufeedbackai_back.git
   cd edufeedbackai_back
   
2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   cd .\venv\Scripts\
   ./activate

3. Instalar las dependencias:
   ```bash
   cd ../..
   pip install -r requirements.txt

4. Aplicar las migraciones:
   ```bash
   python manage.py migrate

5. Crear un superusuario:
   ```bash
   python manage.py createsuperuser

## Crear una API key para la API de Gemini
   
1. Dirigirse a https://aistudio.google.com/app/apikey iniciando sesión con una cuenta de Google. Debe seleccionar un proyecto de Google Cloud, puede ser el que ya viene por defecto llamado "Generative Language Client".
  ![image](https://github.com/user-attachments/assets/b61365e8-a288-4701-94cd-0e9c30c18128)

2. Copiar la API key y dirigirse a la clase "AnalyzeCommentsView" ubicada en el archivo "views.py" de la carpeta "core", y reemplazar el texto "API-KEY" por el valor de la API key. La API key debe ir entre comillas.
  ![image](https://github.com/user-attachments/assets/b4d579b4-9a70-4662-b19a-bbc111c8014a)

## Ejecución

1. Iniciar el servidor de desarrollo:
   ```bash
   python manage.py runserver

2. La API estará disponible en http://127.0.0.1:8000

## Instalar el Frontend de la aplicación

Dirigirse a https://github.com/santa51107HD/edufeedbackai_front

## Notas
1. Se incluye en el repositorio del frontend un archivo en formato Excel para cargar datos de prueba en la aplicación. Este archivo contiene datos ficticios.
2. El conjunto de datos proporcionado por la DACA no incluye un código único para identificar a los docentes, por lo que, temporalmente, se utilizan sus nombres completos como identificador. Posteriormente, a cada nombre completo único se le asigna un ID.
3. Debido a la anonimización de los datos, cada vez que se importe un archivo de prueba en Excel, se crearán nuevos usuarios docentes, ya que no existe un identificador único constante.
4. Si se desea implementar una identificación más consistente para los docentes en la aplicación, se deben realizar los siguientes pasos:
    - Agregar una columna llamada "CEDULA" en el archivo de prueba Excel y completar dicha columna con los datos correspondientes para cada registro.
    - Luego, en el archivo "views.py" dentro de la carpeta "core", dirigirse a la clase "ExcelUploadView", comentar las líneas 164, 168, 357 y 390, y descomentar las líneas 167, 356 y 391.

      ![image](https://github.com/user-attachments/assets/b98a40ea-5fc4-4c4d-96ca-1a2494314363)
      ![image](https://github.com/user-attachments/assets/13409c08-f4fb-423c-bcc7-5dbb5affcca4)
      ![image](https://github.com/user-attachments/assets/7881ffef-72ec-4a5d-a08c-8c61f70a6855)
      ![image](https://github.com/user-attachments/assets/55f404ab-a231-4b76-a2d4-0aedf25e107a)
      
    - Este ajuste permitirá utilizar la cédula como identificador único de los docentes.
