from django.contrib import admin
from .models import Docente, Materia, Comentario, Evaluacion, Usuario, DirectorPrograma, Daca
# Register your models here.
admin.site.register(Usuario)
admin.site.register(DirectorPrograma)
admin.site.register(Daca)
admin.site.register(Docente)
admin.site.register(Materia)
admin.site.register(Comentario)
admin.site.register(Evaluacion)