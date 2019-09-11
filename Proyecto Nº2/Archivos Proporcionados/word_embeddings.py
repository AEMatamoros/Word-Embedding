from os import listdir
from os.path import isfile, join
from gensim.models import Word2Vec
import numpy as np
import os
import unicodedata#Libreria que nos permite eliminar las tildes
import re#Libreria que nos permite utilizar expresiones regulares(NOs ayudara a realizar split delimitado por varios caracteres)


class WordEmbedding:

  def __init__(self, corpus_path):
	  documentos=list(os.listdir(corpus_path))
	  corpus=[]
	  for documento in documentos:
	    abrirDoc= open(corpus_path+"/"+str(documento),"r")
	    texto= abrirDoc.read().lower()#Asignacion de el archivo de texto a una variable de tipo string
	    #Eliminar Acentos
	    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
	    texto = unicodedata.normalize("NFKC", unicodedata.normalize("NFKD", texto).translate(trans_tab))
	    #Separando las palabras y agregandolas a una lista
	    #Cambiar caracteres especiales y numeros por espacios en blanco
	    texto_sin_caracteres = re.sub("[^a-zA-Z0-9 \n\.]", " ", texto).replace("."," ")
	    #eliminar numeros
	    texto_formateado=re.sub(r'\d','',texto_sin_caracteres)
	    #Dividir split (por espacio en blanco) 
	    lista_palabras= texto_formateado.split()
	    #Agregando listas de palabras a corpus
	    corpus.append(lista_palabras)
	  print(corpus)
	  
   
  pass
  """
    El parámetro corpus_path representa la dirección de un directorio que contiene documentos en formato txt.

    Se deberá separar cada documento en palabras, este proceso se hará de forma simplificada de la siguiente forma: convertir todo el texto a minúscula, remover todas las vocales con acentos, separar las palabras asumiendo que son secuencias continuas de caracteres del alfabeto español, cualquier cosa que no sea un caracter (símbolo, espacio en blanco, número) no se considerará como parte de la palabra y servirá para separarla de la siguiente palabra.

    *Es conveniente que al finalizar este método se cree una lista de documentos (conocida como corpus), que contenga la lista de las palabras separadas de cada uno de los documentos txt encontrados en el directorio.

    A manera de ejemplo, si el corpus estuvise formado por los siguientes documentos:

    documento_1:

    La inteligencia artificial (IA), es la inteligencia llevada a cabo por máquinas. En ciencias de la computación, una máquina «inteligente» ideal es un agente flexible que percibe su entorno y lleva a cabo acciones que maximicen sus posibilidades de éxito en algún objetivo o tarea.1​ Coloquialmente, el término inteligencia artificial se aplica cuando una máquina imita las funciones «cognitivas» que los humanos asocian con otras mentes humanas, como por ejemplo: «percibir», «razonar», «aprender» y «resolver problemas».2

    documento_2:

    Andreas Kaplan y Michael Haenlein definen la inteligencia artificial como "la capacidad de un sistema para interpretar correctamente datos externos, para aprender de dichos datos y emplear esos conocimientos para lograr tareas y metas concretas a través de la adaptación flexible".3​ A medida que las máquinas se vuelven cada vez más capaces, tecnología que alguna vez se pensó que requería de inteligencia se elimina de la definición. Por ejemplo, el reconocimiento óptico de caracteres ya no se percibe como un ejemplo de la «inteligencia artificial» habiéndose convertido en una tecnología común.4​ Avances tecnológicos todavía clasificados como inteligencia artificial son los sistemas de conducción autónomos o los capaces de jugar al ajedrez o al Go.5​
    
    Entonces la lista que se debería crear en este método sería:

    [
      ['la', 'inteligencia', 'artificial', 'ia', 'es', 'la', 'inteligencia', 'llevada', 'a', 'cabo', 'por', 'maquinas', 'en', 'ciencias', 'de', 'la', 'computacion', 'una', 'maquina', 'inteligente', 'ideal', 'es', 'un', 'agente', 'flexible', 'que', 'percibe', 'su', 'entorno', 'y', 'lleva', 'a', 'cabo', 'acciones', 'que', 'maximicen', 'sus', 'posibilidades', 'de', 'exito', 'en', 'algun', 'objetivo', 'o', 'tarea', 'coloquialmente', 'el', 'termino', 'inteligencia', 'artificial', 'se', 'aplica', 'cuando', 'una', 'maquina', 'imita', 'las', 'funciones', 'cognitivas', 'que', 'los', 'humanos', 'asocian', 'con', 'otras', 'mentes', 'humanas', 'como', 'por', 'ejemplo', 'percibir', 'razonar', 'aprender', 'y', 'resolver', 'problemas'],

      ['andreas', 'kaplan', 'y', 'michael', 'haenlein', 'definen', 'la', 'inteligencia', 'artificial', 'como', 'la', 'capacidad', 'de', 'un', 'sistema', 'para', 'interpretar', 'correctamente', 'datos', 'externos', 'para', 'aprender', 'de', 'dichos', 'datos', 'y', 'emplear', 'esos', 'conocimientos', 'para', 'lograr', 'tareas', 'y', 'metas', 'concretas', 'a', 'traves', 'de', 'la', 'adaptacion', 'flexible', 'a', 'medida', 'que', 'las', 'maquinas', 'se', 'vuelven', 'cada', 'vez', 'mas', 'capaces', 'tecnologia', 'que', 'alguna', 'vez', 'se', 'penso', 'que', 'requeria', 'de', 'inteligencia', 'se', 'elimina', 'de', 'la', 'definicion', 'por', 'ejemplo', 'el', 'reconocimiento', 'optico', 'de', 'caracteres', 'ya', 'no', 'se', 'percibe', 'como', 'un', 'ejemplo', 'de', 'la', 'inteligencia', 'artificial', 'habiendose', 'convertido', 'en', 'una', 'tecnologia', 'comun', 'avances', 'tecnologicos', 'todavia', 'clasificados', 'como', 'inteligencia', 'artificial', 'son', 'los', 'sistemas', 'de', 'conduccion', 'autonomos', 'o', 'los', 'capaces', 'de', 'jugar', 'al', 'ajedrez', 'o', 'al', 'go']

    ]
    """

  def bag_of_words(self):
    """ 
    Este método retornará una matriz de entrenamiento cuyas columnas representarán cada uno de los documentos del corpus, el tamaño de la matriz será de nxm, donde n es el número total de palabras diferentes que hay en todo el corpus y m es el número de documentos del corpus.
    
    Para formar cada columna de la matriz se seguirán los siguientes pasos:
      1. Determinar n.
      2. Ordenar alfabéticamente las n palabras, de modo que se tenga una posición para cada palabra. 
      3. Cada vector columna representará la cantidad de veces que aparece en cada documento la palabra que corresponde a la posición definida en el paso anterior, si la palabra no aparece la cantidad será 0.

      Por ejemplo, para el corpus definido en el método __init__ la matriz de entrenamiento sería como se muestra en el documento adjunto ejemplo_bag_of_words.txt

    """
    pass


  def tf_idf(self):
    """
    Este método retornará una matriz de entrenamiento similar al método anterior, con la diferencia que esta vez, los valores de la matriz en vez de contener un simple conteo de las palabras que aparecen en el documento contendrán cada uno la estadística tf_idf que se calculará para cada palabra de cada documento de la siguiente forma:

    tf_idf = tf * idf
    tf = Número de veces que aparece la palabra / Total de palabras en el documento
    idf = log_base10( Número total de documentos / Número de documentos que contienen la palabra )

    Por ejemplo, para el corpus definido en el método __init__ la matriz de entrenamiento sería como se muestra en el documento adjunto ejemplo_tf_idf.txt    

    """
    pass

  def word2vec(self):
    """
    Este método utlizará un modelo basado en redes neuronales para determinar la matriz de entrenamiento llamado como word2vec. No es necesario que se cree la red neuronal ni conocer los detalles de implementación del mismo, sino solamente usar la librería adecuada para hacerlo, en este caso la librería a usar se llama gensim y el modelo Word2Vec.

    Sin embargo Word2Vec como su nombre lo indica permite obtener un vector a partir de una palabra, no de un documento como se necesita en este caso. Para obtener dicho vector se utilizará la técnica mencionada en: https://arxiv.org/pdf/1607.00570.pdf, que consiste en los siguientes pasos:

      1. Obtener el vector de atributos de tamaño n para cada una de las palabras del documento, usando el modelo Word2Vec.
      2. Obtener un vector min de tamaño n que contiene los valores mínimos encontrados para cada uno de los n atributos de todas la palabras del documento.
      3. Obtener un vector max de tamaño n que contiene los valores máximos encontrados para cada uno de los n atributos de todas la palabras del documento.
      4. Concatenar el vector min y el vector max en un sólo vector (en ese orden) de tamaño 2n y usar este vector como vector de atributos del documento.
      5. Realizar los pasos 1-4 para todos los documentos del corpus y formar la matriz de entrenamiento.

    Por ejemplo, para el corpus definido en el método __init__ la matriz de entrenamiento sería como se muestra en el documento adjunto ejemplo_word2vec.txt 

    Importante: Para que al autograder funcione correctamente asegúrese de entrenar el modelo con los siguientes parámetros size=50, min_count=1, workers=1, seed=1 y antes de correr el autograder debe haber inicializado la variable de entorno PYTHONHASHSEED=1 (use el comando export de bash)
    """    
    pass
    
    
#ruta=r'/ruta1'#Creacion de la ruta
#print(os.getcwd())#Obtiene el directorio actual
#print(os.listdir())#Imprime archivos y directorios

#Posicionamiento en la carpeta con los archivos de prueba
nuevaruta = str(os.getcwd())+r'/test_corpus' 

#Crear objeto de tipo WordEmbedding
WEmbedding1= WordEmbedding(nuevaruta)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
