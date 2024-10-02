# Generated by Django 5.0.6 on 2024-09-23 21:58

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_alter_evaluacion_comentario_alter_evaluacion_docente_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='DirectorEscuela',
            fields=[
                ('usuario', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, related_name='director_escuela', serialize=False, to=settings.AUTH_USER_MODEL)),
                ('escuela', models.CharField(max_length=50)),
            ],
        ),
        migrations.RenameField(
            model_name='usuario',
            old_name='is_director_programa',
            new_name='is_director_escuela',
        ),
        migrations.DeleteModel(
            name='DirectorPrograma',
        ),
    ]