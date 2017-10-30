# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()


training_data = read_file ( "2015_tme3_usps_train.txt" )
test_data = read_file ( "2015_tme3_usps_test.txt" )

# affichage du 1er chiffre "2" de la base:
#display_image ( training_data[0][0] )
#print training_data[0][0].shape

#print np.sum(training_data[0][0])
#print training_data[0][0]
#affichage du 5ème chiffre "3" de la base:
#display_image ( training_data[3][4] )


def learnML_class_parameters(classarray):
    esperance = np.zeros((256))
    sigma = np.zeros((256))
    cpt = 0
    for image in classarray:
        cpt +=1
        for pixel in range(256):
            esperance[pixel] += image[pixel]
    esperance = esperance/cpt
    for image in classarray:
        for pixel in range(256):
            sigma[pixel] += pow(image[pixel] - esperance[pixel], 2)
    sigma = sigma/cpt
    return np.array([esperance,sigma])


#print learnML_class_parameters(training_data[1])


def learnML_all_parameters(fullarray):
    parmeters = list()
    for iclass in range(10):
        parmeters.append(learnML_class_parameters(fullarray[iclass]))
    return parmeters


parameters = learnML_all_parameters(training_data)


def log_likelihood(number, numparam):
    likelihood = 0
    for i in range(256):
        if numparam[1][i] != 0:
            likelihood += -0.5*math.log(2*math.pi*numparam[1][i]) - \
                          0.5*(pow((number[i]-numparam[0][i]), 2))/numparam[1][i]
    return likelihood

#print log_likelihood(test_data[2][3], parameters[1])
#print [ log_likelihood ( test_data[0][0], parameters[i] ) for i in range ( 10 ) ]

def log_likelihoods(number,parameters):
    res = list()
    for i in range(10):
        res.append(log_likelihood(number,parameters[i]))
    return res

#print log_likelihoods(test_data[1][5], parameters)

def classify_image(number, parameters):
    ll = log_likelihoods(number, parameters)
    tmp = ll[0]
    ind = 0
    for i in range(len(ll)):
        if ll[i] > tmp:
            tmp = ll[i]
            ind = i
    return ind

def classify_all_images(test_data,parameters):
    res = np.zeros((10,10))
    for i in range(len(test_data)):
        tmp = test_data[i]
        for j in range(len(tmp)):
            res[i][classify_image(tmp[j], parameters)] += 1
        res[i] = res[i]/len(tmp)
    return res

test = classify_all_images(test_data,parameters)
print test
plt.imshow(test,interpolation="nearest")