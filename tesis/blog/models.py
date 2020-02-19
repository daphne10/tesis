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
  text = models.TextField(max_length=700)
  emotion = models.CharField(help_text="La emoción que debe seleccionar es, la que el autor cree que esta transmitiendo con el texto ingresado anteriormente",max_length=15, choices=EMOTION, default='joy')
  result = models.TextField(blank=True, null=True)
  res_eval = models.BooleanField(default=True)
  image = models.ImageField(upload_to='images/',blank=True, null=True)
  image_modify = models.ImageField(upload_to='images/',blank=True, null=True)
  joy = models.IntegerField(default=0, blank=True, editable=False)
  trust = models.IntegerField(default=0, blank=True, editable=False)
  fear = models.IntegerField(default=0, blank=True, editable=False)
  surprise = models.IntegerField(default=0, blank=True, editable=False)
  sadness = models.IntegerField(default=0, blank=True, editable=False)
  disgust = models.IntegerField(default=0, blank=True, editable=False)
  anticip = models.IntegerField(default=0, blank=True, editable=False)
  anger = models.IntegerField(default=0, blank=True, editable=False)

