from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken, AuthTokenSerializer
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.settings import api_settings

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
