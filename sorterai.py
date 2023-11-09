import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()



#sample Layer
class Layer_Dense:

    # Layer initialisieren (beim erstellen des Layers)
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # weights und biases zufällige werte
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #regularizer strenght setzen
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # normal output berechnen
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backpropagation der Werte
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradients on regularization
        #put L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        #put L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        #L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases


        self.dinputs = np.dot(dvalues, self.weights.T)


#Dropout

class Layer_Dropout:

    def __init__(self, rate):
        #store and invert rate
        self.rate = 1 - rate

    def forward(self, inputs):

        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


#Activations

#Sigmoid activation (0 min, 1 max)
class Activation_Sigmoid:
    def forward(self,inputs):
        #save inputs of sigmoid
        self.inputs = inputs
        self.output = 1 / (1+ np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

# ReLU activation (rectified linear unit)
class Activation_ReLU:

    # normal berechnen
    def forward(self, inputs):
        self.inputs = inputs
        #wenn größer null = wert, sonst 0
        self.output = np.maximum(0, inputs)

    # backpropagation
    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        # wenn kleiner null = 0
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation (e**x)
class Activation_Softmax:

    # normal berechnen
    def forward(self, inputs):

        self.inputs = inputs


        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # normalizen
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # backpropagation
    def backward(self, dvalues):

        # array erstellen, das die form von dvalues hat
        self.dinputs = np.empty_like(dvalues)

        
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix berechen
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

#Optimizers

# SGD optimizer
class Optimizer_SGD:

    # standartwerte festlegen
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # einmal aufrufen vor optimazation
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # parameter updaten
    def update_params(self, layer):

        # wenn momentum benutzt wird
        if self.momentum:

            # wenn kein array mit momentums, eins mit 0 erstellen
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)

            # weight updaten
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # bias updaten
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        #weights und biases updaten
        layer.weights += weight_updates
        layer.biases += bias_updates


    # iteration erhöhen
    def post_update_params(self):
        self.iterations += 1

# Adagrad optimizer
class Optimizer_Adagrad:

    # werte initialisieren
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # parameter updaten
    def update_params(self, layer):

        #wie bei anderen optimizern:
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)


        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)


    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:
#wie bei anderen optimizern

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho


    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:

    # wie bei anderen optimizern
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)


    def post_update_params(self):
        self.iterations += 1


# generellen Loss berechnen
class Loss:


    def calculate(self, output, y):

        # loss für samples berechnen
        sample_losses = self.forward(output, y)

        # durchschnitt berechnen
        data_loss = np.mean(sample_losses)

        return data_loss
    
    def regularization_loss(self, layer):

        regularization_loss = 0

        #L1 - weights
        #only factor > 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 - bias
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


# Cross-entropy loss funktion
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # länge eines batches = samples
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, dvalues, y_true):

        # anzahl an samples berechnen
        samples = len(dvalues)

        labels = len(dvalues[0])


        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # gradient berechnen
        self.dinputs = -y_true / dvalues
        # gradient normalizen
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        #clip to prevent division by 0
        #clip both sides to prevent mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * 
                          np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of outputs per sample, count with first sample
        outputs = len(dvalues[0])
        #clip to prevent division by 0 and clip both sides to prevent mean
        clipped_values = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_values - (1 - y_true) /
                         (1 - clipped_values)) / outputs
        
        self.dinputs = self.dinputs / samples

#loss for softmax
class Activation_Softmax_Loss_CategoricalCrossentropy():


    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()


    def forward(self, inputs, y_true):

        self.activation.forward(inputs)

        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)


    def backward(self, dvalues, y_true):


        samples = len(dvalues)

        # wenn one-hot, verwandeln
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# Dataset erstellen
#X, y = spiral_data(samples=100, classes=2)
X, y = spiral_data(samples=100, classes=3)

x_tester = np.zeros(100,5)
y_tester = np.zeros(100,5)

# create list in list
# from [0,0,0,0,0,...] to [[0],[0],[0],[0],...]
#y = y.reshape(-1,1)

#layer 1 mit 64 neuronen und 2 inputs
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# activation function ReLU
activation1 = Activation_ReLU()

dropout1 = Layer_Dropout(0.1)
# layer 2 mit 64 neuronen (64 outputs von layer 1) und 3 outputs (3 farben)
dense2 = Layer_Dense(64, 1)

activation2 = Activation_Sigmoid()
# 2. activation mit verbindung zu loss
loss_activation = Loss_CategoricalCrossentropy()

#loss_function = Loss_BinaryCrossentropy()

# optimizer auswählen
#optimizer = Optimizer_SGD(decay=1e-3, momentum = 0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)
#optimizer = Optimizer_RMSprop(decay=1e-4)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)
#optimizer = Optimizer_Adam(decay=5e-7)

# 10001 epochen
for epoch in range(10001):

    #normaler forward pass
    dense1.forward(X)

    activation1.forward(dense1.output)

    #dropout variante
    
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)


    #binary variante
    #dense2.forward(activation1.output)
    #activation2.forward(dense2.output)

    data_loss = loss_activation.forward(dense2.output, y)
    #data_loss = loss_function.calculate(activation2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    #regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Accuracy berechnen
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    #predictions = (activation2.output > 0.5) * 1
    #accuracy = np.mean(predictions==y)


    #jede 100. epoche alle werte anzeigen
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3}, ' +
              f'loss: {loss:.3} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # backward pass ausführen (partial derivatives)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    #loss_function.backward(activation2.output, y)
    #activation2.backward(loss_function.dinputs)
    #dense2.backward(activation2.dinputs)
    #activation1.backward(dense2.dinputs)
    #dense1.backward(activation1.dinputs)


    # optimizer verändert weights und biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
	y_test = np.argmax(y_test, axis= 1) 
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

#X_test, y_test = spiral_data(samples=100, classes=2)
#print(X_test, y_test)
#y_test = y_test.reshape(-1,1)
#dense1.forward(X_test)
#activation1.forward(dense1.output)
#dense2.forward(activation1.output)
#activation2.forward(dense2.output)
#loss = loss_function.calculate(activation2.output, y_test)
#predictions = (activation2.output > 0.5) * 1
#accuracy = np.mean(predictions==y_test)
#print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
