import numpy as np

grade = 100

def end_and_print_grade():
  print('=' * 79)
  if grade == 100:
    print('¡Felicidades no se detectó ningún error!')
  print(('Su nota asignada es: NOTA<<{0}>>').format(grade if grade >= 0 else 0))
  exit()

def print_error_and_exception(error, exception):
  print(error)
  print("La excepcion recibida fue: \n\n{0}\n\n".format(e))

try:
  from word_embeddings import WordEmbedding
except Exception as e:
  print_error_and_exception(
    'No se pudo importar la clase WordEmbedding del módulo word_embeddings', e)
  grade = 0
  end_and_print_grade()

try:
  we = WordEmbedding('test_corpus')
except Exception as e:
  print_error_and_exception('No se pudo crear una instancia de WordEmbedding', e)
  grade = 0
  end_and_print_grade()

try:
  if not np.allclose(we.bag_of_words(), np.load('ans1.npy')):
    print('Error en los resultados del método bag_of_words -25%')
    grade -= 25
except Exception as e:
  print_error_and_exception('Error al invocar el método bag_of_words -25%', e)
  grade -= 25

try:
  if not np.allclose(we.tf_idf(), np.load('ans2.npy')):
    print('Error en los resultados del método tf_idf -35%')
    grade -= 35
except Exception as e:
  print_error_and_exception('Error al invocar el método tf_idf -35%', e)
  grade -= 35

try:
  if not np.allclose(we.word2vec(), np.load('ans3.npy')):
    print(we.word2vec())
    print(np.load('ans3.npy'))
    print('Error en los resultados del método word2vec -40%')
    grade -= 40
except Exception as e:
  print_error_and_exception('Error al invocar el método word2vec -40%', e)
  grade -= 40

end_and_print_grade()