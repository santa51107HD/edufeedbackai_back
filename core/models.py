from django.db import models
from django.contrib.auth.models import (AbstractBaseUser, PermissionsMixin, BaseUserManager)
# Create your models here.

class UserManager(BaseUserManager):
    def create_user(self, id, password, **extra_fields):
        if not id:
            raise ValueError('Falta el ID')
        user = self.model(id=id , **extra_fields)
        user.set_password(password)
        user.save(using=self._db)

        return user
    
    def create_superuser(self, id, password):
        user = self.create_user(id, password)
        user.nombre = id
        user.is_staff = True
        user.is_superuser = True
        user.is_daca = True
        user.save(using=self._db)

        # Crear instancia de Daca asociada al superusuario
        Daca.objects.create(usuario=user)

        return user

class Usuario(AbstractBaseUser, PermissionsMixin):
    id = models.CharField(max_length=20, primary_key=True)
    nombre = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    is_director_escuela = models.BooleanField('director status', default=False)
    is_docente = models.BooleanField('docente status', default=False)
    is_daca = models.BooleanField('daca status', default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False, null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD = 'id'

class DirectorEscuela(models.Model):
    usuario = models.OneToOneField(Usuario, on_delete=models.CASCADE, primary_key=True, related_name='director_escuela')
    escuela = models.CharField(max_length=50)

class Docente(models.Model):
    usuario = models.OneToOneField(Usuario, on_delete=models.CASCADE, primary_key=True, related_name='docente')
    genero = models.CharField(max_length=10)

class Daca(models.Model):
    usuario = models.OneToOneField(Usuario, on_delete=models.CASCADE, primary_key=True, related_name='daca')

class Materia(models.Model):
    codigo = models.CharField(max_length=20, primary_key=True)
    nombre = models.CharField(max_length=200)
    escuela = models.CharField(max_length=50)

class Comentario(models.Model):
    id = models.AutoField(primary_key=True)
    comentario = models.TextField()
    comentario_limpio = models.TextField()
    calificacion = models.IntegerField()
    sentimiento = models.CharField(max_length=3)

class Evaluacion(models.Model):
    comentario = models.ForeignKey(Comentario, on_delete=models.CASCADE, db_column='comentario_id', related_name='evaluaciones_comentario')
    docente = models.ForeignKey(Docente, on_delete=models.CASCADE, db_column='docente_id', related_name='evaluaciones_docente')
    materia = models.ForeignKey(Materia, on_delete=models.CASCADE, db_column='materia_codigo', related_name='evaluaciones_materia')
    grupo = models.IntegerField()
    semestre = models.CharField(max_length=10)
    anho = models.IntegerField()

    class Meta:
        unique_together = (('comentario', 'docente'),)  # Combinación única de comentario y docente