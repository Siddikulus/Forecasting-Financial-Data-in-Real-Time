from django.db import models

# Create your models here.

class SampleData(models.Model):
    id = models.AutoField(primary_key=True)
    Quarter = models.CharField(max_length=2)
    Year = models.CharField(max_length=4)
    feature = models.CharField(max_length = 100)
    value = models.DecimalField(max_digits = 100, decimal_places=2)


class TargetVariable(models.Model):
    id = models.AutoField(primary_key=True)
    Target = models.CharField(max_length = 100)

class CorrelationMatrix(models.Model):
    id = models.AutoField(primary_key=True)
    correlation_from = models.CharField(max_length = 200)
    correlation_to = models.CharField(max_length = 200)
    correlation_coefficient = models.DecimalField(max_digits = 100, decimal_places=2)


