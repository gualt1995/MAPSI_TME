import pickle as pkl
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

fname = "dataVelib.pkl"
f= open(fname,'rb')
data = pkl.load(f)
f.close()
data = np.asarray(data)
#print data


def matricegeo(dat):
    matrixar = []
    for station in dat:
        if station['number']//1000 >= 1 and station['number']//1000 <= 20:
            if len(matrixar) == 0:
                matrixar.append([station['number']//1000,station['bike_stands'],station['available_bike_stands']])
            else:
                added = False
                for i in range(0,len(matrixar)):
                    if matrixar[i][0] == station['number']//1000:
                        matrixar[i] = [station['number']//1000, matrixar[i][1]+station['bike_stands'],matrixar[i][2]+station['available_bike_stands']]
                        added = True
                        break
                if not added:
                    matrixar.append([station['number'] // 1000, station['bike_stands'], station['available_bike_stands']])
    return  matrixar


def probaar(dat):
    """matrixar = [0]*21
    for station in dat:
        if station['number'] // 1000 >= 1 and station['number'] // 1000 <= 20:
            matrixar[station['number']//1000] += 1
    """
    arrondissement = []
    for station in dat:
        arrondissement.append(station['number'])
    res = plt.hist(arrondissement)
    arr = res[1]
    intervalle = arr[1] - arr[0]
    parr = res[0] / res[0].sum()
    parr /= intervalle
    fig = plt.figure(2)
    plt.bar((arr[1:] + arr[:-1]) / 2, parr, arr[1] - arr[0])
    fig.show()
    raw_input()



def probaal(dat):
    altitudes = []
    for station in data:
        altitudes.append(station['alt'])
    nintervalles = 100
    res = plt.hist(altitudes, nintervalles)
    #creation de l'histogramme
    alt = res[1]
    intervalle = alt[1] - alt[0]
    pAlt = res[0] / res[0].sum()
    pAlt /= intervalle
    fig = plt.figure(2)
    plt.bar((alt[1:] + alt[:-1]) / 2, pAlt, alt[1] - alt[0])
    res = (alt[1:] + alt[:-1]) / 2, pAlt, alt[1] - alt[0]
    return res
    # fig.show()
    # raw_input() #pour que les graphes restent


def probaspal(dat):
    tabalt = probaal(dat)
    altitudes = []
    spal = []
    for i in tabalt[1]:
        for station in data:
        if station['available_bike_stands'] == 0:
            spal.append([station['alt'],1])
        else:
            spal.append([station['alt'],0])
    nintervalles = 100
    res = plt.hist(spal, nintervalles)
    return res

def mapsetup(stations):
    x1 = stations[:, 3]
    x2 = stations[:, 2]
    style = [(s, c) for s in "o^+*" for c in "byrmck"]

    plt.figure()
    for i in range(1, 21):
        ind, = np.where(stations[:, 0] == i)
        # scatter c'est plus joli pour ce type d'affichage
        plt.scatter(x1[ind], x2[ind], marker=style[i - 1][0], c=style[i - 1][1], linewidths=0)

    plt.axis('equal')
    plt.legend(range(1, 21), fontsize=10)
    plt.savefig("carteArrondissements.pdf")

print probaal(data)

