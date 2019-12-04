from django.db import models

EMOTION = (
    ('joy','Alegría'),
    ('trust', 'Confianza'),
    ('fear','Miedo'),
    ('surprise','Sorpresa'),
    ('sadness','Tristeza'),
    ('disgust','Asco'),
    ('anticipation','Anticipación'),
    ('anger','Enojo'),
)

class Pln(models.Model):
  text = models.TextField()
  emotion = models.CharField(max_length=15, choices=EMOTION, default='joy')
  result = models.TextField(blank=True, null=True)
  res_eval = models.BooleanField(default=True)
  list_res = models.CharField(max_length=15,blank=True, null=True)
  list_emotion = models.CharField(max_length=15,blank=True, null=True)
  image = models.ImageField(upload_to='images/',blank=True, null=True)
  image_modify = models.ImageField(upload_to='images/',blank=True, null=True)