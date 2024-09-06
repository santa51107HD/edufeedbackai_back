from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken, AuthTokenSerializer
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.settings import api_settings
from rest_framework.views import APIView
import google.generativeai as genai

from .models import Evaluacion
from .serializers import EvaluacionSerializer

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
        comentarios = request.data

        # Configura la API de Gemini
        genai.configure(api_key="API-KEY")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # Prepara el mensaje para Gemini
        mensaje = f"""Quiero que analices la siguiente informacion y me des tu opinion sobre los comentarios positivos,
            neutrales y negativos por cada semestre que hicieron los estudiantes a sus docentes. Quiero que tu opinion 
            sea resumida para no abrumar al usuario con mucha informacion, puedes usar 2000 caracteres como maximo y 
            toda la informacion debe estar en un solo parrafo: {comentarios}"""

        # Genera la respuesta usando Gemini
        response = model.generate_content(mensaje)

        # Retorna la respuesta de Gemini en la respuesta HTTP
        return Response({
            "texto_generado": response.text
        }, status=status.HTTP_200_OK)