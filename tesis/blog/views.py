from django.shortcuts import render
from django.utils import timezone
from .models import Pln
from django.shortcuts import get_object_or_404
from .forms import PlnForm
from django.shortcuts import redirect
import nltk
#from nltk.corpus import treebank
from nltk.corpus import stopwords
from collections import Counter
import re
import pandas as pd
from textblob import TextBlob
#from textblob import Word
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from langdetect import detect
from PIL import Image
import numpy as np
#import os
#import math
import matplotlib.pyplot as plt
#import numpy as np
#import urllib.request
#from tsp_solver.greedy_numpy import solve_tsp
#from scipy.spatial.distance import pdist, squareform 
#from matplotlib.colors import ListedColormap
from django.conf import settings
#from matplotlib.pyplot import *

def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))  
    return pixel

def process_image(image,color_emocion):
    if len(color_emocion) != 0: 
        nt,n,total_per = 50000,0,0
        image_file = image.convert('1', dither=Image.NONE)        
        bw_image_array = np.array(image_file, dtype=np.int)  
        black_indices = np.argwhere(bw_image_array == 0)  
        plt.figure(figsize=(8, 10), dpi=100) 
        colorl = list();
        per = list();
        for j in range(len(color_emocion)):
            colorl.append(color_emocion[j][0])
            per.append(color_emocion[j][1])
        for l in range(len(per)):
            total_per = total_per+(per[l])
        for j in range(len(colorl)):
            n = int((nt * per[j])//total_per)
            chosen_black_indices = black_indices[  
                                   np.random.choice(black_indices.shape[0],  
                                                    replace=False,  
                                                    size=n)]  
            plt.scatter([x[1] for x in chosen_black_indices],
                        [x[0] for x in chosen_black_indices],
                        s = 0.5,
                        c=colorl[j])       
        plt.gca().invert_yaxis()
        plt.xticks([])  
        plt.yticks([])
        plt.savefig('{}/images/result.png'.format(settings.MEDIA_ROOT),bbox_inches='tight')
    else:
      new = convert_dithering(image)
      save_image(new,'{}/images/result.png'.format(settings.MEDIA_ROOT))

def get_saturation(value, quadrant):
  if value > 223:
    return 255
  elif value > 159:
    if quadrant != 1:
      return 255

    return 0
  elif value > 95:
    if quadrant == 0 or quadrant == 3:
      return 255

    return 0
  elif value > 32:
    if quadrant == 1:
      return 255

    return 0
  else:
    return 0

# Create a dithered version of the image
def convert_dithering(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to half tones
  for i in range(0, width, 2):
    for j in range(0, height, 2):
      # Get Pixels
      p1 = get_pixel(image, i, j)
      p2 = get_pixel(image, i, j + 1)
      p3 = get_pixel(image, i + 1, j)
      p4 = get_pixel(image, i + 1, j + 1)

      # Color Saturation by RGB channel
      red   = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
      green = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
      blue  = (p1[2] + p2[2] + p3[2] + p4[2]) / 4

      # Results by channel
      r = [0, 0, 0, 0]
      g = [0, 0, 0, 0]
      b = [0, 0, 0, 0]

      # Get Quadrant Color
      for x in range(0, 4):
        r[x] = get_saturation(red, x)
        g[x] = get_saturation(green, x)
        b[x] = get_saturation(blue, x)

      # Set Dithered Colors
      pixels[i, j]         = (r[0], g[0], b[0])
      pixels[i, j + 1]     = (r[1], g[1], b[1])
      pixels[i + 1, j]     = (r[2], g[2], b[2])
      pixels[i + 1, j + 1] = (r[3], g[3], b[3])

  return new 

def pln_traducir(field):
    if detect(field) != "en":
        blob = TextBlob(field)
        fieldT = str(blob.translate(to='en'))
    else:
        fieldT = field
    
    fieldT = fieldT.lower()
    
    return fieldT

def pln_clean(fieldT):
    output = str(TextBlob(fieldT).correct()) #Corregir texto - pendiente en español
    
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', output) #Limpiar texto
    cleaned = re.sub('[!#?,.:";]', '', cleaned) #Quitar signos de puntuación
    return cleaned   

def pln_tokenizar(cleaned):
    stop_words = list(set(stopwords.words('english'))) #Obtener palabras de eliminación
    tokens = nltk.word_tokenize(cleaned)   
    #Lematizar
    #wordnet_lemmatizer = WordNetLemmatizer()
    #s2 = list()
    #for w in tokens:
    #    s2.append(wordnet_lemmatizer.lemmatize(w))
    
    #Stemm
    #ps =PorterStemmer()
    #s1 = list()
    #for w in tokens:    
    #   rootWord=ps.stem(w)
    #   s1.append(rootWord)   
    stopped = [w for w in tokens if not w in stop_words] #Guarda solo las palabras necesarias (stopped)     
    return stopped
    
def pln_tagged(stopped):        
    tagged = nltk.pos_tag(stopped) #Etiquetas POS
    '''counts = Counter(tag for word,tag in tagged) #Validar si existe verbo,adjetivo y sustantivo
    total = sum(counts.values())            
    a = dict((word, float(count)/total) for word,count in counts.items())'''
    entities = nltk.chunk.ne_chunk(tagged) #Entidades
    return entities

def pln_corpus(): 
    filepath = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t')
    emolex_df.head(12)
    
    emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    emolex_words.head()
    return emolex_words

def pln_emocion(stopped):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    lemma_function = WordNetLemmatizer()
    
    s3 = list()            
    for token, tag in pos_tag(stopped):
        lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
        s3.append(lemma)
    anger, anticipation, disgust,fear,joy,sadness,surprise,trust,total=0,0,0,0,0,0,0,0,0
    #emotion_test = list() 
    emolex_words = pln_corpus()
    for j in range(len(s3)):
    #calificaciones_1 = (emolex_words[emolex_words.word == s3[j]])
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.anger == 1)].empty:
          anger = anger +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.anticipation == 1)].empty:
          anticipation = anticipation +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.disgust == 1)].empty:
          disgust = disgust +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.fear == 1)].empty:
          fear = fear +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.joy == 1)].empty:
          joy = joy +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.sadness == 1)].empty:
          sadness = sadness +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.surprise == 1)].empty:
          surprise = surprise +1   
        if not emolex_words[(emolex_words.word == s3[j]) & (emolex_words.trust == 1)].empty:
          trust = trust +1   
    #print(s3[j])
    #print(calificaciones_1)
    #emotion_test.append(calificaciones_1) 
    total = len(s3)
    list_valores = list()               
    list_valores.append(round((anger/total)*100,2))
    list_valores.append(round((anticipation/total)*100,2))
    list_valores.append(round((disgust/total)*100,2))
    list_valores.append(round((fear/total)*100,2))
    list_valores.append(round((joy/total)*100,2))
    list_valores.append(round((sadness/total)*100,2))
    list_valores.append(round((surprise/total)*100,2))
    list_valores.append(round((trust/total)*100,2))
    return list_valores

def pln_color(list_valores):
    list_colors = list()
    list_colors.append('#D40000') #anger
    list_colors.append('#FF7D00') #anticipation
    list_colors.append('#DE00DE') #disgust
    list_colors.append('#A725DB') #fear
    list_colors.append('#FFE854') #joy
    list_colors.append('#0000C8') #sadness
    list_colors.append('#0089E0') #surprise
    list_colors.append('#00B400') #trust
    color_emocion = list()
    for j in range(len(list_valores)):
        if not list_valores[j] <= 0.0:
            color_emocion.append([list_colors[j],list_valores[j]])
    return color_emocion

def pln_result(request, pk):
    pln = get_object_or_404(Pln, pk=pk)
    return render(request, 'blog/pln_result.html', {'pln': pln})


def pln_new(request):
    if request.method == "POST":
        form = PlnForm(request.POST)
        if form.is_valid():
            list_emociones = list() 
            list_emociones.append('anger')
            list_emociones.append('anticipation')
            list_emociones.append('disgust')
            list_emociones.append('fear')
            list_emociones.append('joy')
            list_emociones.append('sadness')
            list_emociones.append('surprise')
            list_emociones.append('trust')
            #Obtener texto
            pln = form.save(commit=False)
            data = form.cleaned_data
            field = data['text']
            emotion_user = data['emotion']
            #--------------Procesar texto----------------------------
            fieldT = pln_traducir(field)
            cleaned = pln_clean(fieldT)
            #Frecuencia            
            words = nltk.tokenize.word_tokenize(cleaned)
            fd = nltk.FreqDist(words)
            #fd.plot()
            #Sinonimos y definiciones
            #text_word = Word('safe')
            #text_word.definitions
            #synonyms = set()
            #for synset in text_word.synsets:
            #    for lemma in synset.lemmas():
            #        synonyms.add(lemma.name())
            #text_word = Word('safe')

            #antonyms = set()
            #for synset in text_word.synsets:
            #    for lemma in synset.lemmas():
            #        if lemma.antonyms():
            #            antonyms.add(lemma.antonyms()[0].name())

            #print(antonyms)

            #print(synonyms)
            stopped = pln_tokenizar(cleaned)           
            a = pln_tagged(stopped)
            pln.list_emotion = a
            list_valores = pln_emocion(stopped)  

            emotion_esp = ''
            emotion = list_emociones[list_valores.index(max(list_valores))]
            print(list_valores.index(max(list_valores)))
            if list_valores.index(max(list_valores)) == 0:
                emotion = 'neutra'
                emotion_esp = 'Neutra'
            else:               
                if emotion == 'anger':
                    emotion_esp = 'Enojo'
                elif emotion == 'anticipation':
                    emotion_esp = 'Anticipación'
                elif emotion == 'disgust':
                    emotion_esp = 'Asco'
                elif emotion == 'fear':
                    emotion_esp = 'Miedo'
                elif emotion == 'joy':
                    emotion_esp = 'Alegría'
                elif emotion == 'sadness':
                    emotion_esp = 'Tristeza'
                elif emotion == 'surprise':
                    emotion_esp = 'Sorpresa'
                elif emotion == 'trust':
                    emotion_esp = 'Confianza'
                else:
                    emotion_esp = 'Neutra'
            #pln.result = '{}{}{}'.format(emotion,sa_lexicon.process(field),output)
            #pln.result = '{}{}{}{}{}{}'.format(tokens,fieldT,tagged,counts,a,rootWord)
            #pln.result = 'Total palabras: {}, anger: {}, anticipation: {}, disgust: {}, fear: {}, joy: {} , sadness: {}, surprise: {}, trust: {}, Resultado: {}'.format(len(s3),anger,anticipation,disgust,fear,joy,sadness,surprise,trust,list_emociones[list_valores.index(max(list_valores))])
            #zipped_list =list()
            pln.result = '{}'.format(emotion_esp)
            pln.anger = list_valores[0]
            pln.anticip = list_valores[1] 
            pln.disgust = list_valores[2]
            pln.fear = list_valores[3]
            pln.joy = list_valores[4]
            pln.sadness = list_valores[5]
            pln.surprise = list_valores[6]
            pln.trust = list_valores[7]
            #zipped_list == zip(list_emociones, list_valores)     
            if emotion_user == emotion:
                pln.res_eval = True
            else:
                pln.res_eval = False
                
            '''Procesamiento de Imagen'''

            original = open_image('{}/images/{}.png'.format(settings.MEDIA_ROOT,emotion))
            color_emocion = pln_color(list_valores)
            process_image(original,color_emocion) 
       
            #pln.result = '{}{}{}'.format(tokens,stopped,calificaciones_1,list)
            #pln.image = form.cleaned_data['image']

            # = treebank.parsed_sents('wsj_0001.mrg')[0]
            #t.draw() #árbol sintáctico
            pln.image = '/images/{}.png'.format(emotion)
            pln.image_modify = '{}/images/result.png'.format(settings.MEDIA_ROOT)
            pln.save()
            return redirect('pln_result', pk=pln.pk)
    else:
        form = PlnForm()
    return render(request, 'blog/pln_edit.html', {'form': form})