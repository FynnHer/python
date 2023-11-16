import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import os
import cv2
nnfs.init()

#Getting data
def getdata_nichtausfuehren():
    from zipfile import ZipFile
    import os
    import urllib
    import urllib.request

    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)

    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    
    print('Done!')

class Layer_Dense:

    # Layer initialisieren (beim erstellen des Layers)
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # weights und biases zufällige werte
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #regularizer strenght setzen
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # normal output berechnen
    def forward(self, inputs, training):
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

class Layer_Input:
       
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dropout:

    def __init__(self, rate):
        #store and invert rate
        self.rate = 1 - rate

    def forward(self, inputs, training):
        
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

#Activations

#Sigmoid activation (0 min, 1 max)
class Activation_Sigmoid:
    def predictions(self, outputs):
        return(outputs > 0.5) * 1
    
    def forward(self,inputs, training):
        #save inputs of sigmoid
        self.inputs = inputs
        self.output = 1 / (1+ np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

# ReLU activation (rectified linear unit)
class Activation_ReLU:
    def predictions(self, outputs):
        return outputs

    # normal berechnen
    def forward(self, inputs, training):
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

    def predictions(self,outputs):
        return np.argmax(outputs, axis=1)
    # normal berechnen
    def forward(self, inputs, training):

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

# Linear Activation
class Activation_Linear:
    def predictions(self, outputs):
        return outputs

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

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

#Loss

# generellen Loss berechnen
class Loss:
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:    
            #L1 - weights
            #only factor > 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

            # L2 - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

            # L1 - bias
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return regularization_loss
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate_accumulated(self, *, include_regularization=False):
        
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def calculate(self, output, y, *, include_regularization=False):

        # loss für samples berechnen
        sample_losses = self.forward(output, y)

        # durchschnitt berechnen
        data_loss = np.mean(sample_losses)
        
        self.accumulated_sum = np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
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

#loss for softmax
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples

#Binary Cross-entropy loss funktion
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

#loss mean squared error
class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

#loss mean absolute error
class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):
        samples_losses = np.mean(np.abs(y_true - y_pred), axis=1)
        return samples_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

#Accuracys

class Accuracy:
    def calculate(self, predictions, y):
        
        comparisons = self.compare(predictions, y)
        
        accuracy = np.mean(comparisons)
        
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy
    
    def calculate_accumulated(self):
        
        accuracy = self.accumulated_sum / self.accumulated_count
        
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    
    def init(self, y ,reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y , axis=1)
        return predictions == y

class Model:
    
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
                
    def finalize(self):
        #create and set input layer
        self.input_layer = Layer_Input()
        #count all objects
        layer_count = len(self.layers)
        
        self.trainable_layers = []
        
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )
        
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
               self.softmax_classifier_output = \
                   Activation_Softmax_Loss_CategoricalCrossentropy()
           
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1,
              validation_data=None):
        #default value if batch size not set
        
        
        self.accuracy.init(y)
        
        train_steps = 1
        
        if validation_data is not None:
            validation_steps = 1
            
            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
                
        
        for epoch in range(1, epochs+1): 
            
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
                
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size] 
            
                output = self.forward(batch_X, training=True)
                
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                self.backward(output, batch_y)
                
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if not epoch % print_every or step == train_steps -1:
                    print(f'epoch: {epoch}, ' +
                            f'acc: {accuracy:.3f}, ' +
                            f'loss: {loss:.3f} (' +
                            f'data_loss: {data_loss:.3f}, ' + 
                            f'reg_loss: {regularization_loss:.3f}), ' +
                            f'lr {self.optimizer.current_learning_rate}')
                    
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' +
                    f'acc: {epoch_accuracy:.3f}, ' +
                    f'loss: {epoch_loss:.3f} (' +
                    f'data_loss: {epoch_data_loss:.3f}, ' + 
                    f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                    f'lr {self.optimizer.current_learning_rate}')
            
          
                    
            if validation_data is not None:
            
                self.loss.new_pass()
                self.accuracy.new_pass()
                
                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                
                    output = self.forward(batch_X, training=False)
                
                    self.loss.calculate(output, batch_y)
                
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)
                
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                
                print(f'validation, ' +
                    f'acc: {validation_accuracy:.3f}, ' + 
                    f'loss: {validation_loss:.3f}')
                
    def forward(self, X, training):
        
        self.input_layer.forward(X, training)
        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
            
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            
            self.softmax_classifier_output.backward(output, y)
            
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            for layer in reversed(self.layers[:-1]):
                
                layer.backward(layer.next.dinputs)
                
            return
        
        self.loss.backward(output, y)
        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

def oldModel():

    # Dataset erstellen
    #X, y = spiral_data(samples=100, classes=2)
    #X, y = spiral_data(samples=100, classes=3)
    X, y = spiral_data(samples=1000, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)

    #y = y.reshape(-1,1)
    #y_test = y_test.reshape(-1,1)
    #Initialize Model
    model = Model()


    model.add(Layer_Dense(2,512, weight_regularizer_l2=5e-4,
                                bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.1))
    model.add(Layer_Dense(512,3))
    model.add(Activation_Softmax())


    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()

    model.train(X, y, validation_data=(X_test, y_test),
                epochs=10000, print_every=100)
    # optimizer auswählen
    #optimizer = Optimizer_SGD(decay=1e-3, momentum = 0.9)
    #optimizer = Optimizer_Adagrad(decay=1e-4)
    #optimizer = Optimizer_RMSprop(decay=1e-4)
    #optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-5)
    #optimizer = Optimizer_Adam(decay=5e-7)

def load_mnist_dataset(dataset, path):
    
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(
                        path, dataset, label, file
            ), cv2.IMREAD_UNCHANGED)
            
            X.append(image)
            y.append(label)
            
    return np.array(X), np.array(y).astype('uint8')    

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test
#create data
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
#shuffle data



keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#flatten and scale -1 to 1
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -127.5) / 127.5

validation_X = X[25]
validation_y = y[25]

plt.plot(validation_X)

model = Model()

model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,10))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)


plt.plot(validation_X)
