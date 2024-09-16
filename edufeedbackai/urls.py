"""
URL configuration for edufeedbackai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from core.views import EvaluacionesDocenteView,  CreateTokenView, AnalyzeCommentsView, ExcelUploadView, TFIDFView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', CreateTokenView.as_view(), name='login'), #POST
    path('evaluaciones/docente/', EvaluacionesDocenteView.as_view(), name='evaluaciones_docente'), #GET
    path('analizar/comentarios/', AnalyzeCommentsView.as_view(), name='analizar_comentarios'),  # POST
    path('upload-excel/', ExcelUploadView.as_view(), name='upload-excel'), # POST
    path('tfidf-data/', TFIDFView.as_view(), name='tfidf_data'), #GET
]
