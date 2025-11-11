from django.urls import path
from . import views

urlpatterns = [
    # path('', views.index, name='index'),
    path("", views.home, name="home"),   # '/' 에서 바로 랜딩
    path("data/", views.data_utility, name="data"),
    path("api/logs/", views.fetch_logs, name="fetch_logs"),
    path("api/run_dsa_stream/", views.run_dsa_stream, name="run_dsa_stream"),
    path("api/run_dsa/", views.run_dsa_api, name="run_dsa_api"),
    path('api/run_total_stream/', views.run_total_stream, name='run_total_stream'),
    path('api/run_total/', views.run_total, name='run_total'),

    path("network/", views.network_utility, name="network"),
    path("optimization/", views.optimization_utility, name="optimization"),
    path('run/', views.run_automl, name='run_automl'),
    path('total/', views.total_dashboard, name='total_dashboard'),
    path("run_total_pipeline/", views.run_total_pipeline, name="run_total_pipeline"),
    path("download_file/", views.download_file, name="download_file"), 
]
