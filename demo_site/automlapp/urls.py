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
    path("visualize_model_from_structure/", views.visualize_model_from_structure, name="visualize_model_from_structure"),
    path("run_fxp_training/", views.run_fxp_training, name="run_fxp_training"), 

    path("api/dsbn_convert_stream/", views.dsbn_convert_stream, name="dsbn_convert_stream"),
    path("api/dsbn_convert/", views.dsbn_convert_api, name="dsbn_convert_api"),
    path("api/dsbn_train_stream/", views.dsbn_train_stream, name="dsbn_train_stream"),
    path("api/dsbn_train/", views.dsbn_train_api, name="dsbn_train_api"),

    # Network Utility
    # path("network/", views.network_utility, name="network"),
    path("api/network_few_train_stream/", views.network_few_train_stream, name="network_few_train_stream"),
    path("api/network_few_search_stream/", views.network_few_search_stream, name="network_few_search_stream"),
    path("api/network_one_train_stream/", views.network_one_train_stream, name="network_one_train_stream"),
    path("api/network_zero_search_stream/", views.network_zero_search_stream, name="network_zero_search_stream"),
    path("api/network_zero_retrain_stream/", views.network_zero_retrain_stream, name="network_zero_retrain_stream"),

    # Optimization Utility
    # path("optimization/", views.optimization_utility, name="optimization"),
    path("api/opt_fxp_stream/", views.opt_fxp_stream, name="opt_fxp_stream"),
    path("api/opt_loss_train_stream/", views.opt_loss_train_stream, name="opt_loss_train_stream"),
    path("api/opt_loss_custom_stream/", views.opt_loss_custom_stream, name="opt_loss_custom_stream"),
    path("api/opt_mtl_stream/", views.opt_mtl_stream, name="opt_mtl_stream"),




]
