from django.urls import path
from . import views

urlpatterns = [
    path('', views.pln_new, name='pln_new'),
    path('pln/<int:pk>/', views.pln_result, name='pln_result'),
    path('pln/new', views.pln_new, name='pln_new'),
    path('pln/<int:pk>/edit/', views.pln_edit, name='pln_edit'),
]