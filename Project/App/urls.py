from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home),
    path('search/', views.search),
    path('login/', views.loginhandler, name='login'),
    path('', views.register, name='reg'),
    path('movies/',views.Movies),
    path('tv_shows/',views.TvShows)


]
