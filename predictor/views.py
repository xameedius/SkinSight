from django.shortcuts import render, redirect, get_object_or_404
from .ml import predict_pil
from .models import Prediction
from PIL import Image
from .recommendations import get_recommendation_rich
from django.db.models import Q
from .models import Prediction
import csv
from django.http import HttpResponse
from django.utils import timezone
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from .forms import SkinImageForm, SignUpForm, LoginForm
import base64
import uuid
from django.core.files.base import ContentFile
from PIL import Image
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.views.decorators.http import require_POST

def overview(request):
    return render(request, "predictor/overview.html")

def about(request):
    return render(request, "predictor/about.html")

@login_required
def home(request):
    if request.method == "POST":
        form = SkinImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = request.FILES.get("image") or request.FILES.get("mobile_image")
            captured = (form.cleaned_data.get("captured_image") or "").strip()

            # Decide the source file
            if uploaded:
                img_file = uploaded
            else:
                # captured is like "data:image/jpeg;base64,...."
                header, b64data = captured.split(",", 1)
                raw = base64.b64decode(b64data)

                # create a Django file object
                filename = f"webcam_{uuid.uuid4().hex}.jpg"
                img_file = ContentFile(raw, name=filename)

            # Run inference using PIL
            pil_img = Image.open(img_file).convert("RGB")
            result = predict_pil(pil_img)

            rec = get_recommendation_rich(result["label"], result["confidence"])

            pred = Prediction.objects.create(
                user=request.user,
                image=img_file,
                label=result["label"],
                confidence=result["confidence"],
                top3_json=result.get("top3", None),
                urgency=rec["urgency"],
                contagious=rec["contagious"],
                see_doctor=rec["see_doctor"],
                recommendation=rec["summary"],
                self_care_json=rec["self_care"],
                red_flags_json=rec["red_flags"],
            )
            return redirect("result", pred_id=pred.id)
    else:
        form = SkinImageForm()

    return render(request, "predictor/home.html", {"form": form})

@login_required
def result_page(request, pred_id):
    pred = get_object_or_404(Prediction, id=pred_id, user=request.user)
    return render(request, "predictor/result.html", {"pred": pred})

@login_required
def history(request):
    base_qs = Prediction.objects.filter(user=request.user).order_by("-created_at")
    qs = base_qs

    label = (request.GET.get("label") or "").strip()
    urgency = (request.GET.get("urgency") or "").strip().lower()
    doctor = (request.GET.get("doctor") or "").strip().lower()
    q = (request.GET.get("q") or "").strip()

    if label:
        qs = qs.filter(label=label)

    if urgency in {"urgent", "soon", "monitor"}:
        qs = qs.filter(urgency=urgency)

    if doctor == "yes":
        qs = qs.filter(see_doctor=True)
    elif doctor == "no":
        qs = qs.filter(see_doctor=False)

    if q:
        from django.db.models import Q
        qs = qs.filter(
            Q(label__icontains=q) |
            Q(recommendation__icontains=q)
        )

    # ⭐ COUNT BEFORE slicing
    filtered_count = qs.count()
    total_count = base_qs.count()

    items = qs[:200]

    labels = base_qs.values_list("label", flat=True).distinct().order_by("label")

    return render(request, "predictor/history.html", {
        "items": items,
        "labels": labels,

        "selected_label": label,
        "selected_urgency": urgency,
        "selected_doctor": doctor,
        "search_q": q,

        # ⭐ NEW
        "filtered_count": filtered_count,
        "total_count": total_count,
    })

@login_required
def export_history_csv(request):
    """
    Export filtered Prediction history as CSV.
    Respects query params: label, urgency, doctor, q
    """
    qs = Prediction.objects.filter(user=request.user).order_by("-created_at")

    label = (request.GET.get("label") or "").strip()
    urgency = (request.GET.get("urgency") or "").strip().lower()
    doctor = (request.GET.get("doctor") or "").strip().lower()
    q = (request.GET.get("q") or "").strip()

    if label:
        qs = qs.filter(label=label)
    if urgency in {"urgent", "soon", "monitor"}:
        qs = qs.filter(urgency=urgency)
    if doctor == "yes":
        qs = qs.filter(see_doctor=True)
    elif doctor == "no":
        qs = qs.filter(see_doctor=False)
    if q:
        from django.db.models import Q
        qs = qs.filter(Q(label__icontains=q) | Q(recommendation__icontains=q))

    # filename with timestamp
    stamp = timezone.now().strftime("%Y%m%d_%H%M%S")
    filename = f"skinsight_history_{stamp}.csv"

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    writer = csv.writer(response)

    # Header row
    writer.writerow([
        "id", "created_at", "label", "confidence",
        "urgency", "contagious", "see_doctor",
        "recommendation", "self_care", "red_flags",
        "image_url"
    ])

    # Rows
    for p in qs.iterator():
        self_care = " | ".join(p.self_care_json or [])
        red_flags = " | ".join(p.red_flags_json or [])
        image_url = p.image.url if p.image else ""

        writer.writerow([
            p.id,
            p.created_at.isoformat() if p.created_at else "",
            p.label,
            f"{p.confidence:.6f}",
            p.urgency,
            "yes" if p.contagious else "no",
            "yes" if p.see_doctor else "no",
            (p.recommendation or "").replace("\n", " ").strip(),
            self_care,
            red_flags,
            image_url,
        ])

    return response

def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # auto-login after signup
            return redirect("home")
    else:
        form = SignUpForm()
    return render(request, "predictor/signup.html", {"form": form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect("home")
    else:
        form = LoginForm(request)

    return render(request, "predictor/login.html", {"form": form})

@require_POST
def logout_view(request):
    auth_logout(request)
    return redirect("login")