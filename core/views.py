from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken, AuthTokenSerializer
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.settings import api_settings
from rest_framework.views import APIView
import google.generativeai as genai

from .models import Usuario, Docente, Materia, Comentario, Evaluacion
from .serializers import EvaluacionSerializer

import pandas as pd
import gender_guesser.detector as gender
from genderize import Genderize
from django.db.models import Max
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from transformers import pipeline
from tqdm import tqdm
from django.contrib.auth.hashers import make_password
from sklearn.feature_extraction.text import TfidfVectorizer

ERROR_SERIALIZER = "Los datos enviados no son correctos"

# Clase para crear el token
class CreateTokenView(ObtainAuthToken):
    """Create auth token"""
    serializer_class = AuthTokenSerializer
    renderer_classes = api_settings.DEFAULT_RENDERER_CLASSES

     # Metodo para enviar informacion adicional al token
    def post(self, request):
        serializer = self.serializer_class(
            data=request.data, context={'request': request})
        if serializer.is_valid():
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'error': False,
                'token': token.key,
                'id': user.id,
                'nombre': user.nombre,
                'is_director_programa': user.is_director_programa,
                'is_docente': user.is_docente,
                'is_daca': user.is_daca,
            },status=status.HTTP_302_FOUND)
        else:
            return Response({"error": True, "informacion": ERROR_SERIALIZER }, status=status.HTTP_400_BAD_REQUEST)

# Clase para listar las evaluaciones
class EvaluacionesDocenteView(generics.ListAPIView):
    serializer_class = EvaluacionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        usuario = self.request.user
        if usuario.is_docente:
            return Evaluacion.objects.filter(docente__usuario=usuario)
        elif usuario.is_director_programa:
            return Evaluacion.objects.filter(materia__escuela=usuario.director_programa.escuela)
        elif usuario.is_daca:
            return Evaluacion.objects.all()
        return Evaluacion.objects.none()

class AnalyzeCommentsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Recibe el objeto JSON desde el frontend
        datos = request.data.get("datos")
        texto = request.data.get("texto")

        # Configura la API de Gemini
        genai.configure(api_key="API-KEY")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # Prepara el mensaje para Gemini
        mensaje = f"{texto}: {datos}"

        # Genera la respuesta usando Gemini
        response = model.generate_content(mensaje)

        # Retorna la respuesta de Gemini en la respuesta HTTP
        return Response({
            "texto_generado": response.text
        }, status=status.HTTP_200_OK)

# class ExcelUploadView(APIView):
#     permission_classes = [IsAuthenticated, IsAdminUser]

#     def post(self, request):
#         # Verificar si el usuario es admin y autenticado
#         print(request.user)
#         print(request.user.is_staff)
#         # Lógica para manejar el archivo subido
#         return Response({'message': 'Archivo subido correctamente'})
class ExcelUploadView(APIView):
    permission_classes = [IsAuthenticated, IsAdminUser]

    def post(self, request):
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')
        excel_file = request.FILES['file']
        df = pd.read_excel(excel_file)

        """Obtener el nombre de los docentes para estimar su genero"""
        # Con str.split() divido cada palabra en una lista, con str[0] agarro el primer nombre de cada docente, despues pongo todo en minuscula con str.lower()
        #  y hago que la primera letra del nombre sea mayuscula con str.capitalize(). Todo esto para que la libreria gender-guesser funcione correctamente
        df['NOMBRE'] = df['DOCENTEMATERIACODGR'].str.split().str[0].str.lower().str.capitalize()
        # Ahora 'Nombre' contendrá solo el primer nombre de cada persona

        """Usar las librerias gender_guesser y genderize para estimar el genero de los docentes"""
        # Crear una instancia del detector de género de gender_guesser
        detector = gender.Detector()
        # Crear una instancia del detector de género de genderize
        genderize = Genderize()

        """Funcion que detecta el genero segun un nombre usando el detector de genero de gender_guesser"""
        def get_gender(nombre):
            gender = detector.get_gender(nombre)
            if gender == 'mostly_male':
                return 'male'
            elif gender == 'mostly_female':
                return 'female'
            elif gender == 'andy':
                return 'unknown'
            else:
                return gender

        """Crear la columna GENERO en donde irá el genero de los docentes estimado por gender_guesser"""
        df['GENERO'] = df['NOMBRE'].apply(get_gender)

        """Crear un diccionario para los nombres con genero desconocido"""
        nombres_desconocidos = {}

        for column, row in df.iterrows():
            if row['GENERO'] == 'unknown':
                nombres_desconocidos[row['NOMBRE']] = row['GENERO']

        """Utilizar la libreria genderize para encontrar el genero de los nombres que la libreria gender_guesser no pudo estimar"""
        # Iterar sobre los nombres desconocidos y actualizar el género en el diccionario
        for nombre, genero in nombres_desconocidos.items():
            resultado = genderize.get([nombre])
            genero_predicho = resultado[0]['gender']
            nombres_desconocidos[nombre] = genero_predicho

        """Para este caso en especifico quedaron 3 nombres sin genero, se les asigna el genero manualmente"""
        # nombres_desconocidos['Leoviviana'] = 'female'
        # nombres_desconocidos['Johanio'] = 'male'
        # nombres_desconocidos['Johannio'] = 'male'

        """Aplicar los cambios del genero en el dataframe"""
        for nombre, genero in nombres_desconocidos.items():
            df.loc[df['NOMBRE'] == nombre, 'GENERO'] = genero

        """Asignarle un numero identificador a cada docente"""
        # df['DOCENTE_ID'] = pd.factorize(df['DOCENTE'])[0]+1

        """Crear el dataframe df_docente que tendra la informacion del modelo Docente"""
        df_docente = df[['CEDULA', 'GENERO']].drop_duplicates().rename(columns={'CEDULA': 'id','GENERO': 'genero'})

        """Se eliminan las columnas DOCENTEMATERIACODGR y DOCENTE para anonimizar los datos, y tambien se elimina la columna SERIE que no tiene importancia para este estudio"""
        df = df.drop(columns=['SERIE','DOCENTEMATERIACODGR', 'DOCENTE'])

        """Se renombra la columna AÑO que tiene un espacio en blanco innecesario que puedes causar errores"""
        df.rename(columns={'AÑO ': 'AÑO'}, inplace=True)

        """Limpieza de los datos"""
        # Eliminar las filas con valores nulos en la columna 'COMENTARIO'
        df = df.dropna(subset=['COMENTARIO'])

        """Quitar tildes y poner los comentarios en minuscula"""
        df['COMENTARIO'] = df['COMENTARIO'].str.lower()
        reemplazos = {
            'á': 'a',
            'é': 'e',
            'í': 'i',
            'ó': 'o',
            'ú': 'u',
        }
        #Funcion para quitar las tildes de todas las palabras de los comentarios
        def quitar_tildes(texto):
            for original, reemplazo in reemplazos.items():
                texto = texto.replace(original, reemplazo)
            return texto
        df['COMENTARIO'] = df['COMENTARIO'].apply(quitar_tildes)

        """Filtros"""

        """
        WhiteList

        Permitir los comentarios que contengan abjetivos que describan cosas de forma positiva, negativa y neutral y sustantivos que esten asociados a los docentes
        """
        palabras_clave = ['buen', 'bueno', 'buenos', 'buena', 'buenas', "bien", 'mal', 'malo', 'malos', 'mala', 'malas', 'excelente', 'increible', 'terrible','horrible', 'pesimo', 'pesima', 'maravillosa', 'maravilloso', 'regular', 'amamos', 'amor', 'amare', 'mejor', 'mejorar', 'magnifico', 'magnifica', 'entrega', 'entregas', 'entregar', 'profe', 'profesor', 'profesora', 'docente', 'clase' ,'clases', 'curso', 'asignatura', 'materia', 'ambiente', 'nota', 'notas']
        df_filtrado = df[df['COMENTARIO'].str.contains('|'.join(palabras_clave))]

        """
        Black List

        Quitar los comentarios que contengan sustantivos que no esten asociados a los docentes
        """
        palabras_clave = ['silla','sillas', 'mesa', 'mesas', 'aire', 'acondicionado', 'videobeam', "proyector", "proyectores", 'video', 'beam', 'ventilacion', 'ventilador', 'ventiladores', 'frio', 'calor', 'baño', 'baños', 'zancudo', 'zancudos', 'plaga', 'plagas', 'edificio', 'edificios', 'equipo', 'equipos', 'sala', 'salas', 'aula', 'aulas', 'computador', 'computadora', 'computadores']
        df_blacklist = df_filtrado[~df_filtrado['COMENTARIO'].str.contains('|'.join(palabras_clave))]

        """
        Limite de 512 caracteres por comentario

        Solo se podran analizar los comentarios que tengan 512 caracteres o menos, ya que ese es el limite de caracteres que permiten los clasificadores de sentimiento para calcular la polaridad de los comentarios
        """
        df_cortos = df_blacklist[df_blacklist['COMENTARIO'].str.len() <= 512]

        """Se crea un identificador de comentario temporal"""
        # Restablecer los índices del DataFrame
        df_cortos = df_cortos.reset_index(drop=True)
        # Encontrar cual es el id de comentario con mayor valor para seguir con el orden
        max_id_comentario = Comentario.objects.aggregate(Max('id'))['id__max']
        # Insertar la columna ID_COMENTARIO que será el índice de cada comentario
        df_cortos['ID_COMENTARIO'] = df_cortos.index + max_id_comentario + 1

        """Tokenizacion"""
        
        # Tokenizar los comentarios y guardar los resultados en una nueva columna TOKENIZACION
        df_cortos['TOKENIZACION'] = df_cortos['COMENTARIO'].apply(word_tokenize)

        """Stop Words"""
        
        stop_words = set(stopwords.words('spanish'))

        """Quitar las tildes a las stop_words ya que los comentarios no contienen tildes"""
        stopwords_sin_tildes = [quitar_tildes(stopword) for stopword in stop_words]

        """Funcion para eliminar las stopwords de los comentarios tokenizados"""
        def quitar_stop_words(token):
            return [palabra for palabra in token if not palabra in stopwords_sin_tildes]
        # Eliminar las stopwords de los comentarios tokenizados y guardar los resultados en una nueva columna TOKEN_SIN_STOP_WORDS
        df_cortos['TOKEN_SIN_STOP_WORDS'] = df_cortos['TOKENIZACION'].apply(quitar_stop_words)

        """Funcion para eliminar signos de puntuacion, caracteres especiales y numeros"""
        def limpiar_token(token):
            return re.sub(r'[^\w]|[\d]', '', token)
        # Eliminar signos de puntuacion, caracteres especiales y numeros, y guardar los resultados en una nueva columna TOKEN_LIMPIO
        df_cortos['TOKEN_LIMPIO'] = df_cortos['TOKEN_SIN_STOP_WORDS'].apply(lambda tokens: list(filter(None, [limpiar_token(token) for token in tokens])))

        """Eliminar las columnas TOKENIZACION y TOKEN_SIN_STOP_WORDS ya que ya no son necesarias"""
        df_cortos = df_cortos.drop(columns=['TOKENIZACION','TOKEN_SIN_STOP_WORDS'])

        """Crear la columna COMENTARIO_LIMPIO donde iran los comentarios sin stopwords, signos de puntuacion, caracteres especiales y numeros"""
        df_cortos['COMENTARIO_LIMPIO'] = df_cortos['TOKEN_LIMPIO'].apply(lambda x: ' '.join(x))

        """Analisis de Sentimientos"""

        """Importar los clasificadores que definiran el sentimiento/polaridad de los comentarios"""
        classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
        classifier5 = pipeline("text-classification", model="citizenlab/twitter-xlm-roberta-base-sentiment-finetunned")
        classifier9 = pipeline("text-classification", model="pysentimiento/robertuito-sentiment-analysis")

        """Crear una lista con todos los comentarios"""
        comentarios = df_cortos['COMENTARIO'].tolist()

        """Arrays en donde se guardan los resultados de cada clasificador"""
        experto1 = []
        experto2 = []
        experto3 = []

        """Para cada comentario los 3 clasificadores daran su resultado y se guardará en su array correspondiente"""
        for comentario in tqdm(comentarios):
            result = classifier(comentario)[0]
            result2 = classifier5(comentario)[0]
            result3 = classifier9(comentario)[0]

            experto1.append(result['label'])
            experto2.append(result2['label'])
            experto3.append(result3['label'])

        """
        El experto1 es el unico clasificador que da sus resultados en calificacion de estrellas
        por lo que es escogido para que califique los comentarios en una puntuacion de 1 a 5
        para dar una calificacion numerica adicional a la polaridad
        """
        estrellas = experto1

        """Cambiar la puntuacion de estrellas por una calificacion numerica"""
        estrellas = ['1' if x == '1 star' else x for x in estrellas]
        estrellas = ['2' if x == '2 stars' else x for x in estrellas]
        estrellas = ['3' if x == '3 stars' else x for x in estrellas]
        estrellas = ['4' if x == '4 stars' else x for x in estrellas]
        estrellas = ['5' if x == '5 stars' else x for x in estrellas]

        """Cambiar la puntuacion de estrellas por la polaridad"""
        experto1 = ['NEG' if x == '1 star' or x == '2 stars' else x for x in experto1]
        experto1 = ['NEU' if x == '3 stars' else x for x in experto1]
        experto1 = ['POS' if x == '4 stars' or x == '5 stars' else x for x in experto1]

        """Cambiar el nombre de las etiquetas de polaridad"""
        experto2 = ['NEG' if x == 'Negative' else x for x in experto3]
        experto2 = ['NEU' if x == 'Neutral' else x for x in experto3]
        experto2 = ['POS' if x == 'Positive' else x for x in experto3]

        """Funcion que compara la polaridad calculada por cada clasificador y llega a un veredicto final"""
        def comparar_listas(lista1, lista2, lista3):
            nueva_lista = []
            for i in range(len(lista1)):
                if lista1[i] == lista2[i] or lista1[i] == lista3[i]:
                    nueva_lista.append(lista1[i])
                elif lista2[i] == lista3[i]:
                    nueva_lista.append(lista2[i])
                elif lista1[i] == lista2[i] == lista3[i]:
                    nueva_lista.append(lista1[i])
                else:
                    nueva_lista.append("NEU")
            return nueva_lista

        """Calcular la polaridad definitiva de los comentarios"""
        sentimiento = comparar_listas(experto1, experto2, experto3)

        """Crear una nueva columna CALIFICACION que tendra la calificacion numerica de los comentarios"""
        df_cortos['CALIFICACION'] = estrellas

        """Crear una nueva columna SENTIMIENTO que tendra el sentimiento/polaridad de los comentarios"""
        df_cortos['SENTIMIENTO'] = sentimiento

        """Reemplazar los nombres de materias que tienen 2 nombres diferentes por errores de sintaxis"""
        df_cortos['MATERIA'] = df_cortos['MATERIA'].replace('metodologia DE LA INVESTIGACIÓN', 'METODOLOGÍA DE LA INVESTIGACIÓN')
        df_cortos['MATERIA'] = df_cortos['MATERIA'].replace('PRODUCCIÓN mas LIMPIA', 'PRODUCCIÓN MÁS LIMPIA')
        df_cortos['MATERIA'] = df_cortos['MATERIA'].replace('INTRODUCCIÓN A LA INGENIERÍA INGENIERÍA QUÍMICA', 'INTRODUCCIÓN A LA INGENIERÍA -INGENIERÍA QUÍMICA')
        df_cortos['MATERIA'] = df_cortos['MATERIA'].replace('GESTIÓN DE PROYECTOS  ESTADÍSTICA', 'GESTIÓN DE PROYECTOS - ESTADÍSTICA')
        df_cortos['MATERIA'] = df_cortos['MATERIA'].replace('metodologia DE LA INVESTIGACIÓN EN ALIMENTOS', 'METODOLOGÍA DE LA INVESTIGACIÓN EN ALIMENTOS')

        """Crear el dataframe df_materia que tendrá la informacion del modelo Materia"""
        df_materia = df_cortos[['CODIGOMATERIA', 'MATERIA', 'ESCUELA']].drop_duplicates().rename(columns={'CODIGOMATERIA': 'codigo', 'MATERIA': 'nombre', 'ESCUELA': 'escuela'})

        """Crear el dataframe df_comentario que tendrá la informacion del modelo Comentario"""
        print(df_cortos)
        df_comentario = df_cortos[['ID_COMENTARIO','COMENTARIO', 'COMENTARIO_LIMPIO', 'CALIFICACION', 'SENTIMIENTO']].rename(columns={'ID_COMENTARIO': 'id_comentario','COMENTARIO': 'comentario', 'COMENTARIO_LIMPIO': 'comentario_limpio', 'CALIFICACION': 'calificacion', 'SENTIMIENTO': 'sentimiento'})

        """Crear el dataframe df_evaluacion que tendrá la informacion del modelo Evaluacion"""
        df_evaluacion = df_cortos[['GRUPO', 'SEMESTRE', 'AÑO', 'ID_COMENTARIO', 'CODIGOMATERIA', 'CEDULA' ]].rename(columns={'CEDULA': 'docente_id','CODIGOMATERIA': 'materia_codigo', 'GRUPO': 'grupo', 'SEMESTRE':'semestre', 'AÑO': 'anho', 'ID_COMENTARIO': 'comentario_id'})

        """Crear el dataframe df_daca que tendrá la informacion del modelo Daca"""
        # data = {'es_superuser':[True],"id": ['daca'], "nombre":['daca'], 'es_director':[False], 'es_docente':[False], 'es_daca':[True], 'es_activo':[True], 'es_staff':[True]}
        # df_daca = pd.DataFrame(data)

        """Encontrar cuales son todas las escuelas"""
        # escuelas = df_cortos['ESCUELA'].unique()

        """Crear el dataframe df_director que tendrá la informacion del modelo Director"""
        # data = {
        #     "id_director": ['dquimica', 'dambiente', 'dmecanica', 'delec', 'dalimentos', 'dcivil', 'dsistemas', 'dindustria', 'dmaterial', 'destadist'],  # Datos para la columna "ID"
        #     "escuela": escuelas  # Datos para la columna "Escuela"
        # }
        # df_director = pd.DataFrame(data)

        """Crear el dataframe df_director_usuario que tendrá la informacion del modelo Usuario"""
        # df_director_usuario = df_director
        # df_director_usuario['nombre']=df_director_usuario['id_director']
        # df_director_usuario = df_director_usuario.rename(columns={'id_director':'id'})
        # df_director_usuario["es_director"] = True
        # df_director_usuario["es_docente"] = False
        # df_director_usuario["es_daca"] = False
        # df_director_usuario["es_activo"] = True
        # df_director_usuario["es_staff"] = False

        # df_director_usuario = df_director_usuario.drop(columns=["escuela"])
        # df_director_usuario.insert(0, "es_superuser", False)

        """Crear el dataframe df_docente_usuario que tendrá la informacion del modelo Usuario"""
        df_docente_usuario = df_cortos[['CEDULA', 'NOMBRE']].drop_duplicates().rename(columns={'CEDULA': 'id', 'NOMBRE':'nombre'})
        df_docente_usuario["es_director"] = False
        df_docente_usuario["es_docente"] = True
        df_docente_usuario["es_daca"] = False
        df_docente_usuario["es_activo"] = True
        df_docente_usuario["es_staff"] = False

        df_docente_usuario.insert(0, "es_superuser", False)

        """Concatenar los dataframes df_docente_usuario, df_director_usuario y df_daca para insertar la informacion en el modelo Usuario"""
        # df_concatenado = pd.concat([df_docente_usuario, df_director_usuario, df_daca], ignore_index=True)

        """Agregar una columna que tendrá el valor de la contraseña de los usuarios. La contraseña tendrá el mismo valor que el id"""
        df_docente_usuario.insert(3, "contrasenha", df_docente_usuario["id"])

        """Cifrar las contraseñas"""
        df_docente_usuario['contrasenha'] = df_docente_usuario['contrasenha'].apply(lambda x: make_password(str(x)))

        print("ANTES DE FOR 1")

        for _, row in df_docente_usuario.iterrows():
            # Manejo del modelo Usuario
            if not Usuario.objects.filter(id=row['id']).exists():
                Usuario.objects.create(
                    id=row['id'],
                    nombre=row['nombre'],
                    password=row['contrasenha'], 
                    is_docente=row['es_docente'],
                    is_director_programa=row['es_director'],
                    is_daca=row['es_daca']
                )

        print("ANTES DE FOR 2")

        for _, row in df_docente.iterrows():
            usuario_obj = Usuario.objects.get(id=row['id'])
            if not Docente.objects.filter(usuario=usuario_obj).exists():
                Docente.objects.create(
                    usuario=usuario_obj,
                    genero=row['genero']
                )
        
        print("ANTES DE FOR 3")

        for _, row in df_materia.iterrows():
            if not Materia.objects.filter(codigo=row['codigo']).exists():
                Materia.objects.create(
                    codigo=row['codigo'],
                    nombre=row['nombre'],
                    escuela=row['escuela']
                )

        print("ANTES DE FOR 4")

        for _, row in df_comentario.iterrows():
            Comentario.objects.create(
                id=row['id_comentario'],
                comentario=row['comentario'],
                comentario_limpio=row['comentario_limpio'],
                calificacion=row['calificacion'],
                sentimiento=row['sentimiento']
            )

        print("ANTES DE FOR 5")

        for _, row in df_evaluacion.iterrows():
            comentario_obj = Comentario.objects.get(id=row['comentario_id'])
            docente_obj = Docente.objects.get(usuario=row['docente_id'])
            materia_obj = Materia.objects.get(codigo=row['materia_codigo'])
            Evaluacion.objects.create(
                comentario=comentario_obj,
                docente=docente_obj,
                materia=materia_obj,
                grupo=row['grupo'],
                semestre=row['semestre'],
                anho=row['anho']
            )

        return Response({"status": "success"}, status=status.HTTP_201_CREATED)
    
class TFIDFView(APIView):
    permission_classes = [IsAuthenticated]


    def get_filtered_comments(self):
        usuario = self.request.user
        
        # Si el usuario es docente, filtramos evaluaciones por docente y luego obtenemos los comentarios
        if usuario.is_docente:
            evaluaciones = Evaluacion.objects.filter(docente__usuario=usuario)
        
        # Si el usuario es director de programa, filtramos evaluaciones por escuela de la materia
        elif usuario.is_director_programa:
            evaluaciones = Evaluacion.objects.filter(materia__escuela=usuario.director_programa.escuela)
        
        # Si el usuario es daca, obtenemos todas las evaluaciones.
        elif usuario.is_daca:
            evaluaciones = Evaluacion.objects.all()
        
        # Si no es ninguno de los roles, regresamos un queryset vacío
        else:
            evaluaciones = Evaluacion.objects.none()

        # Extraemos los comentarios relacionados a las evaluaciones filtradas
        comentarios = Comentario.objects.filter(evaluaciones_comentario__in=evaluaciones)
        
        return comentarios

    def get(self, request, *args, **kwargs):
        # Obtener los comentarios filtrados según el tipo de usuario
        comentarios = self.get_filtered_comments()

        # Manejar caso sin comentarios
        if not comentarios.exists():
            return Response({"error": "No hay comentarios disponibles."}, status=404)

         # Extraer el campo 'comentario_limpio' de los comentarios
        df = pd.DataFrame(comentarios.values_list('comentario_limpio', flat=True), columns=['comentario_limpio'])
        
        # Calcular TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['comentario_limpio'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        term_importance = tfidf_df.mean().sort_values(ascending=False)
        top_n_tokens = term_importance.head(10).to_dict()  # Ajusta el número de palabras según sea necesario

        return Response(top_n_tokens)