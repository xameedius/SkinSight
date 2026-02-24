from django.db import models
from django.conf import settings

class Prediction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
    )
    
    image = models.ImageField(upload_to="uploads/%Y/%m/%d/")
    label = models.CharField(max_length=200)
    confidence = models.FloatField()
    top3_json = models.JSONField(null=True, blank=True)

    # Rich recommendations (NEW)
    urgency = models.CharField(max_length=10, default="monitor")  # urgent|soon|monitor
    contagious = models.BooleanField(default=False)
    see_doctor = models.BooleanField(default=False)
    recommendation = models.TextField(blank=True, default="")
    self_care_json = models.JSONField(null=True, blank=True)     # list[str]
    red_flags_json = models.JSONField(null=True, blank=True)     # list[str]

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence:.3f}) @ {self.created_at}"