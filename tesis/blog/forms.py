from django import forms
import nltk
from .models import Pln
from langdetect import detect
from textblob import TextBlob
from nltk import pos_tag
from collections import Counter
from nltk.corpus import stopwords
from charsleft_widget.widgets import CharsLeftInput


def pln_entity(field):
    #stop_words = list(set(stopwords.words('english'))) #Obtener palabras de eliminación
    tokens = nltk.word_tokenize(field) 
    #stopped = [w for w in tokens if not w in stop_words]
    stopped = [w for w in tokens]
    tagged = nltk.pos_tag(stopped)
    counts = Counter(tag for word,tag in tagged)
    nn = 0 #sustantivo
    vb = 0 #verbo
    jj = 0 #adjetivo (jj) o adverbio(rb)
    ent = [0,0,0]
    for v in counts:
     if nn<=0 or jj<=0 or vb<=0:
        if v[:2] == 'NN':
            ent[0] = nn+1
        if v[:2] == 'VB':
            ent[1] = vb+1
        if v[:2] == 'JJ' or v[:2] == 'RB':
            ent[2] = jj+1
     else:
        break
    return ent

def pln_traducir(field):
    if detect(field) != "en":
        blob = TextBlob(field)
        try:
            fieldT = str(blob.translate(to='en'))
        except:
            raise forms.ValidationError('El texto no puede ser procesado, por favor ingrese otras palabras')
    else:
        fieldT = field   
    fieldT = fieldT.lower()
    
    return fieldT

class PlnForm(forms.ModelForm): 
    #text = forms.CharField( widget=forms.Textarea )
 
    def __init__( self, *args, **kwargs ):
           super(PlnForm, self).__init__( *args, **kwargs )
           self.fields['text'].label = False
           self.some_label = 'Seleccione la emoción '
           self.fields['emotion'].label = self.some_label
            
    class Meta:
        model = Pln
        fields = ('text','emotion') 
        widgets = {
            'text': forms.Textarea(attrs={'maxlength':700,'placeholder':"Ingrese el texto aquí",}),
        }
            
    def clean(self): 
        super(PlnForm, self).clean() 
        text = self.cleaned_data.get('text')
        if text is not None:
            if len(text) <10: 
                raise forms.ValidationError('El texto debe contener como mínimo 10 Caracteres ')
            ent = pln_entity(pln_traducir(text)) #<!--validaciones-->
            if ent[0] <=0:
                raise forms.ValidationError('El texto debe contener como minímo un Sustantivo')
            if ent[1] <=0:
                raise forms.ValidationError('El texto debe contener como minímo un Verbo')            
            if ent[2] <=0:
                raise forms.ValidationError('El texto debe contener como minímo un Adjetivo o un Adverbio')
        return self.cleaned_data 

