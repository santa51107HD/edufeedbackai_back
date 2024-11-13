from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import DirectorEscuela, Docente, Daca, Materia, Comentario, Evaluacion

class AuthTokenSerializer(serializers.Serializer):
    id = serializers.CharField()
    password = serializers.CharField(
        style={'input_type': 'password'},
        trim_whitespace=False
    )

class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ['id', 'password','nombre', 'is_director_escuela', 'is_docente', 'is_daca']

class DirectorEscuelaSerializer(serializers.ModelSerializer):
    class Meta:
        model = DirectorEscuela
        fields = ['usuario', 'escuela']

class DocenteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Docente
        fields = ['usuario', 'genero']

class DacaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Daca
        fields = ['usuario']

class MateriaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Materia
        fields = ['codigo', 'nombre', 'escuela']

class ComentarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comentario
        fields = ['id', 'comentario', 'comentario_limpio', 'calificacion', 'sentimiento']

class EvaluacionSerializer(serializers.ModelSerializer):
    comentario = ComentarioSerializer()
    docente = DocenteSerializer()
    materia = MateriaSerializer()
    
    class Meta:
        model = Evaluacion
        fields = ['comentario', 'docente', 'materia', 'grupo', 'semestre', 'anho']
