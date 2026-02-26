from django.urls import path
from .views import (
    home,
    history,
    result_page,
    export_history_csv,
    signup,
    login_view,
    logout_view,
    overview,
    about,
    export_result_csv,
)

urlpatterns = [
    path("", overview, name="overview"),  
    path("scan/", home, name="home"),      
    path("about/", about, name="about"),

    path("result/<int:pred_id>/", result_page, name="result"),
    path("result/<int:pred_id>/export.csv", export_result_csv, name="export_result_csv"),

    path("history/", history, name="history"),
    path("history/export.csv", export_history_csv, name="export_history_csv"),

    path("signup/", signup, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
]