from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


BASE_INPUT_CLASS = (
    "w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm "
    "text-slate-900 shadow-sm placeholder:text-slate-400 "
    "focus:border-slate-400 focus:outline-none"
)

BASE_LABEL_CLASS = "text-sm font-semibold text-slate-700"


class SkinImageForm(forms.Form):
    # Normal upload OR mobile camera
    image = forms.ImageField(required=False)

    # Webcam capture (base64 data URL)
    captured_image = forms.CharField(required=False, widget=forms.HiddenInput())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # optional: keep image input hidden because you use a styled label button
        self.fields["image"].widget.attrs.update({"class": "hidden", "id": "uploadInput"})
        self.fields["image"].label = ""

    def clean(self):
        cleaned = super().clean()
        img = cleaned.get("image")
        cap = (cleaned.get("captured_image") or "").strip()

        if not img and not cap:
            raise forms.ValidationError("Please upload an image or capture one using the camera.")
        return cleaned


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tailwind input styling
        for name, field in self.fields.items():
            field.widget.attrs.update({"class": BASE_INPUT_CLASS})

        # Nice placeholders (optional but feels premium)
        self.fields["username"].widget.attrs.update({"placeholder": "Choose a username"})
        self.fields["email"].widget.attrs.update({"placeholder": "Email (optional)"})
        self.fields["password1"].widget.attrs.update({"placeholder": "Create a password"})
        self.fields["password2"].widget.attrs.update({"placeholder": "Confirm password"})


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, field in self.fields.items():
            field.widget.attrs.update({"class": BASE_INPUT_CLASS})

        # Placeholders (optional)
        self.fields["username"].widget.attrs.update({"placeholder": "Username"})
        self.fields["password"].widget.attrs.update({"placeholder": "Password"})