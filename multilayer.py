import numpy as np
#example for call ml = Multilayer(n=[5, 3, 2], fact="tanh")
#This would mean layer 0 has 5 neurons, layer 1 has 3 neurons, and layer 2 has 2 neurons. total 3 capas
#fact es la funcion que se va usar con las neuronas de las capas ocultas y salida
#It is a reference to the object itself, like 'this' in python is self.
class Multilayer:
    def __init__(self, n, fact="sigmoid"):
        """
        n: lista con el número de neuronas por capa (incluye entrada y salida)
        fact: función de activación ("sigmoid", "relu", "linear", "tanh")
        """

        # 1) Número de capas
        self.L = len(n)

        # 2) Neuronas por capa
        self.n = n

        # 3) Campos (h) y activaciones (xi)
        self.h = [np.zeros(n[l]) for l in range(self.L)]#l vectores
        self.xi = [np.zeros(n[l]) for l in range(self.L)]#l vectores

        # 4) Pesos (w) y umbrales (theta), revisar pesos porque es el siguiente punto
        self.w = [None]    # capa 0 no tiene pesos
        self.theta = [None]

        for l in range(1, self.L):
            self.w.append(np.random.randn(n[l], n[l-1]))   # matriz pesos
            self.theta.append(np.random.randn(n[l]))        # vector umbrales

        # 5) Delta para backpropagation
        self.delta = [np.zeros(n[l]) for l in range(self.L)]

        # 6) Cambios de pesos (d_w)            
        self.d_w = [None]
         # 7) Cambios de umbrales (d_theta)
        self.d_theta = [None]
        # 8) Cambios previos (d_w_prev)
        self.d_w_prev = [None]
        # 9) Cambios previos umbrales (d_theta_prev)
        self.d_theta_prev = [None]
        #Add the values by layer, L(layer number)
        #example with example layer 1, neuronas 2 y maximo 3 capas L
        for l in range(1, self.L):
            #fill with zero files, columns 1 to 
            #pending review TO DO
            self.d_w.append()
            #self.d_theta.append()
            #self.d_w_prev.append()
            #self.d_theta_prev.append()

        # 10) Función de activación usada en la entrada
        self.fact = fact
