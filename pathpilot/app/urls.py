from django.urls import path
from . import views

app_name = 'fraud_detection'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_fraud, name='predict'),
    path('dashboard/', views.analytics_dashboard, name='dashboard'),
    path('job/<int:job_id>/', views.job_detail, name='job_detail'),
]
