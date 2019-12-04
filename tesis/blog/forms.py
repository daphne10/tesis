from django import forms

from .models import Pln

class PlnForm(forms.ModelForm):
    def __init__( self, *args, **kwargs ):
           super(PlnForm, self).__init__( *args, **kwargs )
           self.fields['text'].label = False
           self.some_label = 'Seleccione la emoción '
           self.fields['emotion'].label = self.some_label

            
    class Meta:
        model = Pln
        fields = ('text','emotion')
