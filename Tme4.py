from math import *
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def read_file(filename):

    infile = open(filename, "r")
    for ligne in infile:
        if ligne.find("eruptions waiting") != -1:
            break

    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [float(x) for x in ligne.split()]
        data.append(eruption)
        data.append(waiting)
    infile.close()

    # transformation de la liste en tableau 2D
    data = np.asarray(data)
    data.shape = (int(data.size / 2), 2)

    return data


data = read_file("2015_tme4_faithful.txt")


def normale_bidim(x, z, parameters):
    val1 = 1 / (2 * math.pi * parameters[2] * parameters[3] * math.sqrt(1 - math.pow(parameters[4], 2)))
    val2 = -1 / (2 * (1 - math.pow(parameters[4], 2)))
    val3 = math.pow((x - parameters[0]) / parameters[2], 2)
    val4 = 2 * parameters[4] * ((x - parameters[0]) * (z - parameters[1])) / (parameters[2] * parameters[3])
    val5 = math.pow((z - parameters[1]) / parameters[3], 2)
    val6 = np.exp(val2 * (val3 - val4 + val5))
    res = val1 * val6
    return res


#print(normale_bidim(1, 2, (1.0, 2.0, 3.0, 4.0, 0)))
#print(normale_bidim(1, 0, (1.0, 2.0, 1.0, 2.0, 0.7)))


def dessine_1_normale(params):
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    x = np.linspace(x_min, x_max, 100)
    z = np.linspace(z_min, z_max, 100)
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy()
    for i in range(x.shape[0]):
        for j in range(z.shape[0]):
            norm[i, j] = normale_bidim(x[i], z[j], params)

    # affichage
    fig = plt.figure()
    plt.contour(X, Z, norm, cmap=cm.autumn)
    plt.show()


#dessine_1_normale((-3.0, -5.0, 3.0, 2.0, 0.7))

#dessine_1_normale((-3.0, -5.0, 3.0, 2.0, 0.2))


def dessine_normales(data, params, weights, bounds, ax):
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    nb_x = nb_z = 100
    x = np.linspace(x_min, x_max, nb_x)
    z = np.linspace(z_min, z_max, nb_z)
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm0[j, i] = normale_bidim(x[i], z[j], params[0])  # * weights[0]
    norm1 = np.zeros((nb_x, nb_z))
    for j in range(nb_z):
        for i in range(nb_x):
            norm1[j, i] = normale_bidim(x[i], z[j], params[1])  # * weights[1]

    # affichages des normales et des points du dataset
    ax.contour(X, Z, norm0, cmap=cm.winter, alpha=0.5)
    ax.contour(X, Z, norm1, cmap=cm.autumn, alpha=0.5)
    for point in data:
        ax.plot(point[0], point[1], 'k+')


def find_bounds(data, params):
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min(mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:, 0].min())
    x_max = max(mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:, 0].max())
    z_min = min(mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:, 1].min())
    z_max = max(mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:, 1].max())

    return (x_min, x_max, z_min, z_max)


mean1 = data[:, 0].mean()
mean2 = data[:, 1].mean()
std1 = data[:, 0].std()
std2 = data[:, 1].std()

params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                   (mean1 + 0.2, mean2 + 1, std1, std2, 0)])
weights = np.array([0.4, 0.6])
bounds = find_bounds(data, params)

# affichage de la figure
#fig = plt.figure()
#ax = fig.add_subplot(111)
#dessine_normales(data, params, weights, bounds, ax)
#plt.show()


def Q_i(data, current_params, current_weights):
    res = np.empty((len(data), 2))
    cpt = 0
    for i in data:
        val0 = current_weights[0] * normale_bidim(i[0], i[1], current_params[0])
        val1 = current_weights[1] * normale_bidim(i[0], i[1], current_params[1])
        res[cpt] = ([val0 / (val0 + val1), val1 / (val1 + val0)])
        cpt = cpt + 1
    return res

current_params = np.array([[3.28778309, 69.89705882, 1.13927121, 13.56996002, 0.],
                           [3.68778309, 71.89705882, 1.13927121, 13.56996002, 0.]])

current_weights = np.array([0.5, 0.5])

#T = Q_i(data, current_params, current_weights)


# print(T)

def M_step(data, Q, current_params, current_weights):
    cpt = 0
    sumQ_Y0 = 0
    sumQ_Y1 = 0
    sumQ_Y0_Y1 = 0
    sumQ_Y0_times_Xi = 0
    sumQ_Y1_times_Xi = 0
    sumQ_Y0_times_Zi = 0
    sumQ_Y1_times_Zi = 0

    for i in Q:
        sumQ_Y0 = sumQ_Y0 + i[0]
        sumQ_Y1 = sumQ_Y1 + i[1]
        sumQ_Y0_Y1 = sumQ_Y0_Y1 + i[0] + i[1]

        sumQ_Y0_times_Xi = sumQ_Y0_times_Xi + i[0] * data[cpt][0]
        sumQ_Y1_times_Xi = sumQ_Y1_times_Xi + i[1] * data[cpt][0]
        sumQ_Y0_times_Zi = sumQ_Y0_times_Zi + i[0] * data[cpt][1]
        sumQ_Y1_times_Zi = sumQ_Y1_times_Zi + i[1] * data[cpt][1]

        cpt = cpt + 1

    pi_0 = sumQ_Y0 / sumQ_Y0_Y1
    pi_1 = sumQ_Y1 / sumQ_Y0_Y1

    mu_X0 = sumQ_Y0_times_Xi / sumQ_Y0
    mu_X1 = sumQ_Y1_times_Xi / sumQ_Y1

    mu_Z0 = sumQ_Y0_times_Zi / sumQ_Y0
    mu_Z1 = sumQ_Y1_times_Zi / sumQ_Y1

    cpt = 0
    sigma_X0_tmp = 0
    sigma_X1_tmp = 0
    sigma_Z0_tmp = 0
    sigma_Z1_tmp = 0

    for i in Q:
        sigma_X0_tmp = sigma_X0_tmp + i[0] * math.pow((data[cpt][0] - mu_X0), 2)
        sigma_X1_tmp = sigma_X1_tmp + i[1] * math.pow((data[cpt][0] - mu_X1), 2)
        sigma_Z0_tmp = sigma_Z0_tmp + i[0] * math.pow((data[cpt][1] - mu_Z0), 2)
        sigma_Z1_tmp = sigma_Z1_tmp + i[1] * math.pow((data[cpt][1] - mu_Z1), 2)
        cpt = cpt + 1

    sigma_X0 = math.sqrt(sigma_X0_tmp / sumQ_Y0)
    sigma_X1 = math.sqrt(sigma_X1_tmp / sumQ_Y1)
    sigma_Z0 = math.sqrt(sigma_Z0_tmp / sumQ_Y0)
    sigma_Z1 = math.sqrt(sigma_Z1_tmp / sumQ_Y1)

    cpt = 0
    p0_tmp = 0
    p1_tmp = 0

    for i in Q:
        p0_tmp = p0_tmp + i[0] * ((data[cpt][0] - mu_X0) * (data[cpt][1] - mu_Z0) / (sigma_X0 * sigma_Z0))
        p1_tmp = p1_tmp + i[1] * ((data[cpt][0] - mu_X1) * (data[cpt][1] - mu_Z1) / (sigma_X1 * sigma_Z1))
        cpt = cpt + 1

    p0 = p0_tmp / sumQ_Y0
    p1 = p1_tmp / sumQ_Y1
    current_params = array([(mu_X0, mu_Z0, sigma_X0, sigma_Z0, p0), (mu_X1, mu_Z1, sigma_X1, sigma_Z1, p1)])
    current_weights = array([pi_0, pi_1])
    return (current_params, current_weights)


current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485, 79.76680985, 0.52047055, 7.04450242, 0.58358284)])
current_weights = array([0.45165145, 0.54834855])
#Q = Q_i(data, current_params, current_weights)
#print(M_step(data, Q, current_params, current_weights))



mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1 = data[:,0].std ()
std2 = data[:,1].std ()
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

#Q = Q_i(data, current_params, current_weights)
#print(M_step(data, Q, current_params, current_weights))


def em(steps, data, current_params, current_weights):
    RES = []
    for step in range(steps):
        Q = Q_i(data, current_params, current_weights)
        M = M_step(data, Q, current_params, current_weights)
        RES.append(M)
        current_params = M[0]
        current_weights = M[1]
        #fig = plt.figure()
        #bounds = find_bounds(data, current_params)
        #dessine_normales(data, current_params, current_weights, bounds, fig.add_subplot(111))
        #plt.show()
    return RES



#em(10, data, params, weights)

def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds


res_EM = em(20, data, params, weights)
bounds = find_video_bounds ( data, res_EM )


fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))

def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print "step animate = %d" % ( i )

anim = animation.FuncAnimation(fig, animate,
                               frames = len ( res_EM ), interval=500 )
plt.show ()