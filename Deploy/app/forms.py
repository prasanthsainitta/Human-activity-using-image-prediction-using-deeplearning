from django import forms
from .models import UserImageModel

class UserImageForm(forms.ModelForm):
    class Meta():
        model = UserImageModel
        fields = ['image']