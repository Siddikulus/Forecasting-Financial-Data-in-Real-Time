from django.contrib import admin
from .models import SampleData, TargetVariable, CorrelationMatrix


# Register your models here.

admin.site.register(SampleData)
admin.site.register(TargetVariable)
admin.site.register(CorrelationMatrix)