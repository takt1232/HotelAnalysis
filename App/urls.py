from django.urls import path
from . import views

urlpatterns =[
    path("", views.index, name="index"),
    path("index", views.index, name="index"),
    path('upload/', views.process_csv, name='upload_csv'),
    path('test_model/', views.test_model, name='test_model')
]
