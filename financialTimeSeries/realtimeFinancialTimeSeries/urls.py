from django.contrib import admin
from django.urls import include, path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("selecttarget", views.select_target_variable, name="select_target"),
    path("eda", views.eda, name="eda"),
    path("correlation", views.correlation, name="correlation"),
    path("modelselection", views.modelselection, name="modelselection"),
    path("evaluation", views.evaluation, name="evaluation"),

]