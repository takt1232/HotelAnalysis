from django.urls import path
from . import views

urlpatterns =[
    path("", views.index, name="index"),
    path("index", views.index, name="home"),
    path('upload/', views.process_csv, name='upload_csv'),
    path('test_model/', views.test_model, name='test_model'),
    path('about/', views.about, name='about'),
    path('test/', views.test_view, name='test')
]
