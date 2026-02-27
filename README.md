# SkinSight — LJMU BSCS Final Project
**Creator:** Sameed Ahmed  
**University:** Liverpool John Moores University (LJMU)  
**Project Type:** Final Year Project (BSCS)

SkinSight is an AI-powered web application that analyzes skin images and predicts likely skin condition categories using deep learning.

It provides:
- Predicted label + confidence score
- Urgency badge (urgent / soon / monitor)
- Contagious badge (when relevant)
- “See a doctor” yes/no guidance
- Self-care tips + red flags
- Private user accounts (each user sees their own history)
- History filtering + CSV export

> ⚠️ **Disclaimer:** This project is a prototype and NOT medical advice. Always consult a licensed clinician if you are concerned.

---

# What SkinSight Does

- SkinSight allows users to upload a skin image (or capture one using webcam / mobile camera) and runs a deep learning model to identify a likely condition category.

- The application demonstrates a complete end-to-end machine learning pipeline:

- Dataset → Model Training → Model Inference → Web Interface → User History → Exportable Results

- It is designed as an academic prototype for AI-assisted medical triage support.
 
---

# Technical Specifications

## Backend
- Framework: Django (Python)
- Authentication: Django Auth (login / signup / logout)
- Database: SQLite (default Django database)
- History storage per user
- Filtering and CSV export of predictions

## Machine Learning
- Framework: PyTorch
- Model library: timm
- Base model: EfficientNet-B0
- Input resolution: 224×224
- Dataset loading: torchvision ImageFolder
- Training split: train / validation
- Model artifacts saved in:
  - artifacts/skinsight_model.pt
  - artifacts/class_names.json
  - artifacts/model_meta.json

## Frontend
- Tailwind CSS (CDN)
- Django template rendering
- Laptop webcam capture
- Mobile camera capture
- Image preview system
- Progress animation
- Confidence donut chart
- Top prediction bar charts

---


## Setup Instructions (Clone & Run)

# Setup Instructions (Clone & Run)

# 1 Clone the repository

- git clone https://github.com/xameedius/SkinSight.git
- cd SkinSight
- git clone https://github.com/xameedius/SkinSight.git
- cd SkinSight

# 2 Create and activate a virtual environment
python -m venv .venv
- .\.venv\Scripts\Activate.ps1

# 3 Install dependencies
- pip install -r requirements.txt

# 4 Apply database migrations
- python manage.py migrate

# 5 Run the development server
- python manage.py runserver

- Open browser:
- http://127.0.0.1:8000/

---

## Features Checklist

- ✔ Upload image (desktop / mobile gallery)
- ✔ Mobile camera capture
- ✔ Laptop webcam capture
- ✔ Prediction label + confidence
- ✔ Urgency classification
- ✔ Contagious detection flag
- ✔ Doctor recommendation
- ✔ Self-care guidance
- ✔ Red flag warnings
- ✔ User authentication
- ✔ Private scan history
- ✔ History filtering
- ✔ CSV export
- ✔ Modern responsive UI

---

## Academic Use

- This project was developed as an LJMU BSCS Final Year Project prototype.

- If reused or referenced, please cite:

- Sameed Ahmed — SkinSight (LJMU BSCS Final Project)

---

## License

- Educational and research use only.
- Not intended for medical diagnosis.
