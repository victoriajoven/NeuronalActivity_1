import numpy as np

# 
# CARGA DE DATOS PREPROCESADOS


X_train = np.load("data/processed/X_train_preprocessed.npy")
X_test  = np.load("data/processed/X_test_preprocessed.npy")

y_train = np.load("data/learn/y_train.npy")
y_test  = np.load("data/check/y_test.npy")

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# FUNCIONES DE ACTIVACIÓN

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    # x es la ACTIVACIÓN, no el campo
    return x * (1 - x)



# MULTILAYER CON BACKPROPAGATION + MOMENTUM

class Multilayer:
    def __init__(self, n, eta=0.05, alpha=0.5):
        """
        n: lista con neuronas por capa (ej: [4, 6, 1])
        eta: learning rate
        alpha: momentum
        """

        self.n = n
        self.L = len(n)
        self.eta = eta
        self.alpha = alpha

        # Activación fija = sigmoide
        self.f = sigmoid
        self.df = d_sigmoid

        # Campos, activaciones y deltas
        self.h = [np.zeros(n[l]) for l in range(self.L)]
        self.xi = [np.zeros(n[l]) for l in range(self.L)]
        self.delta = [np.zeros(n[l]) for l in range(self.L)]

        # Pesos y umbrales
        self.w = [None]
        self.theta = [None]

        for l in range(1, self.L):
            self.w.append(np.random.randn(n[l], n[l-1]) * 0.1)
            self.theta.append(np.random.randn(n[l]) * 0.1)

        # Momentum
        self.d_w_prev = [None] + [np.zeros_like(self.w[l]) for l in range(1, self.L)]
        self.d_theta_prev = [None] + [np.zeros_like(self.theta[l]) for l in range(1, self.L)]


    # FEED-FORWARD, added to multilayer review document G
    def feed_forward(self, x):
        self.xi[0] = x

        for l in range(1, self.L):
            self.h[l] = self.w[l] @ self.xi[l-1] - self.theta[l]
            self.xi[l] = self.f(self.h[l])

        return self.xi[-1]


    # BACKPROPAGATION,, added to multilayer review document G
    def backprop(self, target):
        # Salida (ecuación 11)
        self.delta[-1] = self.df(self.xi[-1]) * (self.xi[-1] - target)

        # Capas ocultas (ecuación 12)
        for l in range(self.L-2, 0, -1):
            self.delta[l] = self.df(self.xi[l]) * (self.w[l+1].T @ self.delta[l+1])


    # ACTUALIZACIÓN PESOS, by requiment in document 
    def update(self):

     for l in range(1, self.L):

        # d_w tiene la misma forma que w[l]: (n[l], n[l-1])
        d_w = np.zeros_like(self.w[l])
        d_theta = np.zeros_like(self.theta[l])

        
        # Actualización PESO POR PESO
        
        for i in range(self.n[l]):          # neurona destino en capa l
            for j in range(self.n[l-1]):    # neurona origen capa l-1

                # ecuación (14) — EXACTAMENTE lo del documento
                d_w[i, j] = (
                    -self.eta * self.delta[l][i] * self.xi[l-1][j]
                    + self.alpha * self.d_w_prev[l][i, j]
                )

        
        # Actualización de umbrales
        
        for i in range(self.n[l]):
            d_theta[i] = (
                self.eta * self.delta[l][i]
                + self.alpha * self.d_theta_prev[l][i]
            )

        
        # APLICAR LAS ACTUALIZACIONES
        self.w[l] += d_w
        self.theta[l] += d_theta

        # Guardar cambios para momentum
        self.d_w_prev[l] = d_w
        self.d_theta_prev[l] = d_theta



    # ENTRENAMIENTO COMPLETO
    def train(self, X, y, epochs=2000):

        for epoch in range(epochs):

            # Entrenamiento online (patrón a patrón)
            for i in range(len(X)):
                self.feed_forward(X[i])
                self.backprop(y[i])
                self.update()

            # Mostrar error parcial
            if epoch % 200 == 0:
                print("Epoch", epoch, "Error:", self.error(X, y))


    # Calcular error
    def error(self, X, y):
        E = 0
        for i in range(len(X)):
            pred = self.feed_forward(X[i])
            E += np.sum((pred - y[i])**2)
        return E / 2


# CREAR Y ENTRENAR LA RED NEURONAL

# Tu dataset tiene 4 entradas y 1 salida CO2
n_input = X_train.shape[1]

ml = Multilayer(
    n=[n_input, 6, 1],  # 4 , 6 ocultas y 1 salida
    eta=0.07,
    alpha=0.4
)

ml.train(X_train, y_train, epochs=2000)


# EVALUACIÓN EN TEST
predictions = np.array([ml.feed_forward(x) for x in X_test]).flatten()
mse = np.mean((predictions - y_test.flatten())**2)

print("\nMSE sobre test:", mse)


# EJEMPLO DE PREDICCIÓN

nuevo = X_test[0]  # un vector cualquiera ya preprocesado
pred = ml.feed_forward(nuevo)

print("\nPredicción CO2 (normalizada):", pred)
print("Valor real (normalizado):", y_test[0])
#Se añade neuronalNet con porcentaje de training como valor de entrada. 
class NeuralNet:
    def __init__(self, X, y, percentage, layers=[6], eta=0.05, alpha=0.5):
        """
        percentage: porcentaje de datos usados para TRAINING.
                    Si percentage = 0 → 100% training.
        """

        if percentage == 0:
            # TODO training
            self.X_train = X
            self.y_train = y
            self.X_val = None
            self.y_val = None

        else:
            # Si el porcentaje es un número tipo 75, lo pasamos a 0.75
            if percentage > 1:
                p = percentage / 100.0
            else:
                p = percentage

            N = len(X)
            N_train = int(N * p)

            self.X_train = X[:N_train]
            self.y_train = y[:N_train]
            self.X_val  = X[N_train:]
            self.y_val  = y[N_train:]

        # Arquitectura dinámica según las dimensiones de entrada/salida
        n_input = X.shape[1]
        n_output = y.shape[1]
        architecture = [n_input] + layers + [n_output]

        self.mlp = Multilayer(
            n=architecture,
            eta=eta,
            alpha=alpha
        )
