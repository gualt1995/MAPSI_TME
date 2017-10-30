import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D


def bernoulli(p) :
    if p>1 or p<0:
        print "valeur incorrecte"
        return
    if np.random.random_sample() > p:
        return 1
    else:
        return 0


def binomiale(n,p):
    return np.random.binomial(n,p)


def histobinomiale(n):
    array = np.random.binomial(n, 0.5, 1000)
    plt.hist(array, n)
    plt.show()


def normale(k, sigma):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    else:
        x = np.linspace(-2*sigma, 2*sigma, num=k)
        y =[]
        for i in x:
            tmp1 = 1/(math.sqrt(2*math.pi)*sigma)
            tmp2 = math.exp(-0.5*math.pow(i/sigma, 2))
            y.append(tmp1*tmp2)
        plt.figure()
        plt.plot(x,y)
        plt.show()
        return y


def proba_affine(k, slope):
    if k % 2 == 0:
        raise ValueError('le nombre k doit etre impair')
    if abs(slope) > 2. / (k * k):
        raise ValueError('la pente est trop raide : pente max = ' +
        str(2. / (k * k)))
    x = np.linspace(0, k-1, num=k)
    y = []
    for i in x:
        y.append((1./k)+(i-((k-1.)/2.))*slope)
    cpt = 0
    for i in y:
        cpt += i
    print cpt
    plt.figure()
    plt.plot(x, y)
    plt.show()
    return  y


def pxy(PA, PB):
    test = np.zeros((len(PA), len(PB)))
    for i in range(len(PA)):
        test[i]= PA[i]*PB
    return test

PA = np.array ( [0.2, 0.7, 0.1] )
PB = np.array ( [0.4, 0.4, 0.2, 0.1] )


def dessine(P_jointe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3, 3, P_jointe.shape[0])
    y = np.linspace(-3, 3, P_jointe.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show()


P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])

def pyz(Pxyzt):
    P_YZ=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            P_YZ[i][j]=np.sum(Pxyzt[:,i,j,:])
    return P_YZ



def pxtcondyz(Pxyzt,Pyz):
    pxtcyz = np.zeros((2,2,2,2))
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    pxtcyz[x][y][z][t] = Pxyzt[x][y][z][t] / Pyz[y][z]
    return pxtcyz


def pxcondyz(Pxtcyz):
    Pxcyz=np.zeros((2, 2, 2))
    ta = np.zeros((2, 2, 2))
    tb = np.zeros((2, 2, 2))
    for x in range(2):
        for y in range(2):
            for z in range(2):
                ta[x][y][z] = Pxtcyz[x][y][z][0]
                tb[x][y][z] = Pxtcyz[x][y][z][1]

    for x in range(2):
        for y in range(2):
            for z in range(2):
                Pxcyz[x][y][z] += Pxtcyz[x][y][z][0] + Pxtcyz[x][y][z][1]

    return Pxcyz



def ptcondyz(Pxtcyz):
    Ptcyz=np.zeros((2, 2, 2))
    ta = np.zeros((2, 2, 2))
    tb = np.zeros((2, 2, 2))
    for y in range(2):
        for z in range(2):
            for t in range(2):
                ta[y][z][t] = Pxtcyz[0][y][z][t]
                tb[y][z][t] = Pxtcyz[1][y][z][t]

    for y in range(2):
        for z in range(2):
            for t in range(2):
                Ptcyz[y][z][t] += Pxtcyz[0][y][z][t] + Pxtcyz[1][y][z][t]

    return Ptcyz

def verif():
    Pverif = np.zeros((2, 2, 2, 2))
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    Pverif[x][y][z][t] += pxcondyz(pxtcondyz(P_XYZT, pyz(P_XYZT)))[x][y][z] * \
                                          ptcondyz(pxtcondyz(P_XYZT, pyz(P_XYZT)))[y][z][t]
    return Pverif



#print pxtcondyz(P_XYZT,pyz(P_XYZT))
#print 'toto'
# pxcondyz(pxtcondyz(P_XYZT,pyz(P_XYZT)))
# ptcondyz(pxtcondyz(P_XYZT,pyz(P_XYZT)))
#print pxcondyz(pxtcondyz(P_XYZT,pyz(P_XYZT))) * ptcondyz(pxtcondyz(P_XYZT,pyz(P_XYZT)))



def verif2():
    Pxyz = np.zeros((2, 2, 2))
    ta = np.zeros((2,2,2))
    tb = np.zeros((2,2,2))

    for x in range(2):
        for y in range(2):
            for z in range(2):
                ta[x][y][z] = P_XYZT[x][y][z][0]
                tb[x][y][z] = P_XYZT[x][y][z][1]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                Pxyz[i][j][k] += ta[i][j][k] + tb[i][j][k]
    #----------------------------------------------------
    Px = np.zeros((2))
    for x in range(2):
        ty = 0
        for y in range(2):
            tz = 0
            for z in range(2):
                tz += Pxyz[x][y][z]
            ty += tz
        Px[x] += ty
    #----------------------------------------------------
    Pyz = np.zeros((2, 2))

    for y in range(2):
        for z in range(2):
            tx = 0
            for x in range(2):
                tx += Pxyz[x][y][z]
            Pyz[y][z] += tx
    #---------------------------------------------------
    Pverif2 = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            Pverif2[0][i][j] = Pyz[0][j] * Px[i]
            Pverif2[1][i][j] = Pyz[1][j] * Px[i]

    #print("Pyz", Pyz)
    #print("Px", Px)
    print("Pxyz", Pxyz)
    print("Pverif2",Pverif2)

verif2()














# ------------------------------------------------------------------------------
# INDEPENDANCES CONDITIONNELLES ET CONSOMATION MEMOIRE
def read_file(filename):




    Pjointe = gum.Potential()
    variables = []

    fic = open(filename, 'r')
    # on rajoute les variables dans le potentiel
    nb_vars = int(fic.readline())
    for i in range(nb_vars):
        name, domsize = fic.readline().split()
        variable = gum.LabelizedVariable(name, name, int(domsize))
        variables.append(variable)
        Pjointe.add(variable)

    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append(float(line))
    Pjointe.fillWith(np.array(cpt))

    fic.close()
    return np.array(variables), Pjointe


print("\n \n")


def conditional_indep(P_jointe, X, Y, Z, eps):


    liste = [X.name(), Y.name()]

    for i in Z:
        liste.append(i.name())
    P_XYZ = P_jointe.margSumIn(liste)

    P_XZ = P_XYZ.margSumOut([Y.name()])
    P_Z = P_XYZ.margSumOut([X.name(), Y.name()])
    P_XcondZ = P_XZ / P_Z
    P_YZ = P_XYZ.margSumOut([X.name()])
    P_YcondZ = P_YZ / P_Z

    Q_XYcondZ = P_XYZ - (P_XcondZ * P_YcondZ)

    if (Q_XYcondZ.abs().max() < eps):
        return True
    else:
        return False


def compact_conditional_proba(P_jointe, X, eps):

    P_S = P_jointe.variablesSequence()
    K = []
    for i in P_S:
        K.append(i)

    for x in K:
        if (conditional_indep(P_jointe, X, x, P_jointe.margSumOut([x.name()]).variablesSequence(), eps)):
            K.remove(x)

    liste_K = []
    for k in K:
        liste_K.append(k.name())
    return P_jointe.margSumIn(liste_K)


def create_bayesian_network(P_jointe, eps):

    liste = []
    P = P_jointe
    liste_P = P.variablesSequence()
    for i in range(len(liste_P) - 1, 0, -1):
        Q = compact_conditional_proba(P, liste_P[i], eps)
        liste_Q = Q.variablesSequence()
        liste_var_Q = []
        for q in liste_Q:
            liste_var_Q.append(q.name())
        # print(liste_var_Q)
        liste.append(Q)
        P.margSumOut([liste_P[i].name()])
    return liste


var, P_jointe = read_file("asia.txt")

l = create_bayesian_network(P_jointe, 0.0001)
print (l)"""
