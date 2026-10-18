from django.urls import path
from .views import *
app_name = 'analyzer'

urlpatterns = [
    path('',index_view,name ='home'),
    path('about/',about_view,name ='about')
]
