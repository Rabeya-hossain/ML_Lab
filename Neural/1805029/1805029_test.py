import numpy as np
import torchvision.datasets as ds
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import f1_score,confusion_matrix
import seaborn as sns

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate,y_true):
        # TODO: update parameters and return input gradient
        pass
    def get_weights(self):
        pass
    def set_weights(self,weights):
        pass
    def get_bias(self):
        pass
    def set_bias(self,bias):   
        pass

class Initializer:
    def __init__(self, stddev_calculation_function, name):
        self.stddev_calculation_function = stddev_calculation_function
        self.name = name
    def __str__(self) -> str:
        return "Initializer: " + self.name
    def __call__(self, input_size, output_size):
        np.random.seed(0)
        stdev = self.calculate_stdev(input_size, output_size)
        return np.random.normal(loc=0, scale=stdev, size=(input_size, output_size))

class HeInitializer(Initializer):
    def __init__(self):
        super().__init__(self.calculate_stdev, "He")
    def calculate_stdev(self, input_size, output_size):
        return np.sqrt(2 / (input_size+output_size))
    
class XavierInitializer(Initializer):
    def __init__(self):
        super().__init__(self.calculate_stdev, "Xavier")
    def calculate_stdev(self, input_size, output_size):
        return np.sqrt(2.0 / (input_size+output_size))

class Adam:
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(self, w, gradient_wrt_w):
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(gradient_wrt_w))
            self.v = np.zeros(np.shape(gradient_wrt_w))
        self.m = self.b1 * self.m + (1 - self.b1) * gradient_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * gradient_wrt_w ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        # print("using adam optimizer")
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def update_bias(self, b, gradient_wrt_b):
        # self.t += 1
        if self.m_b is None:
            self.m_b = np.zeros(np.shape(b))
            self.v_b = np.zeros(np.shape(b))
        self.m_b = self.b1*self.m_b + (1-self.b1)*np.sum(gradient_wrt_b, axis=1, keepdims=True)
        self.v_b = self.b2*self.v_b + (1-self.b2)*np.power(np.sum(gradient_wrt_b, axis=1, keepdims=True), 2)
        m_hat_b = self.m_b/(1-np.power(self.b1, self.t))
        v_hat_b = self.v_b/(1-np.power(self.b2, self.t))
        # print("in bias update")
        # print(b.shape)

        # self.bias -= self.learning_rate*m_hat_b/(np.sqrt(v_hat_b)+1e-8)
        return b - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate,y_true):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Softmax(Layer):
    def forward(self, input):
        if np.isnan(input).any():
            print("nan value in softmax layer input")
        # print("input shape: ", input.shape)
        # print("input: ", input)
        exp_values = np.exp(input - np.max(input, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities
        return self.output
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate, y_true):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        # return output_gradient
        return self.output - y_true
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x < 0, 0 , np.where(x==0, 0.5, 1))
            return np.greater(x, 0).astype(int)

        super().__init__(relu, relu_prime)

class Dropout(Layer):
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, input, training=True):
        if not training:
            return input
        self.mask = np.random.binomial(1, self.p, size=input.shape) / self.p
        return np.multiply(input, self.mask)

    def backward(self, output_gradient, learning_rate, y_true):
        return np.multiply(output_gradient, self.mask)

class Dense(Layer):
    def __init__(self, input_size, output_size,Initializer,Optimizer):
        self.weights = Initializer(output_size, input_size)
        self.bias = Initializer(output_size, 1)
        self.optimizer = Optimizer
        # self.weights = np.random.randn(output_size, input_size)
        # self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate,y_true):
        # implement mini-batch gradient descent here
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        # self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        self.weights = self.optimizer.update(self.weights,weights_gradient)
        self.bias = self.optimizer.update_bias(self.bias,output_gradient)

        return input_gradient
    
    def get_weights(self):
        return self.weights
    def set_weights(self,weights):
        self.weights = weights
    
    def get_bias(self):
        return self.bias
    def set_bias(self,bias):
        self.bias = bias

class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    def cross_entropy(y_true, y_pred):
    # avoid any log(0) calculation
        epsilon = 1e-10  # Small constant to avoid logarithm of zero
        predictions = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted probabilities

        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(predictions + epsilon))
        return loss

    def cross_entropy_prime(y_true, y_pred):
        return -y_true / (y_pred + 1e-8)

class FNN:
    def __init__(self,network,loss,loss_prime,batch_size=1024):
        self.network = network
        self.loss = loss
        self.loss_prime = loss_prime
        self.epochs = 3
        self.batch_size = batch_size
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []
        self.train_F1 = []
        self.validation_F1 = []
        self.output = None
        self.y_true = None
        self.accuracy = None

    def predict(self,input):
        output = input
        
        for layer in self.network:
            # if layer is dropout layer
            if isinstance(layer, Dropout):
                output = layer.forward(output, training=False)
            else:
                output = layer.forward(output)
            
        return output
    
    def calculate(self,epoch,x_validation,y_validation,x_train,y_train):
        # calculate the accuracy of the model in vectorized form
        correct = 0
        validation_prediction = np.zeros((26,len(x_validation)))
        train_prediction = np.zeros((26,len(x_train)))
        for j in range(0, len(x_validation), self.batch_size):
            # forward
            output = x_validation[j:j+self.batch_size]
            output= output.T
            new_y = y_validation[j:j+self.batch_size]
            new_y = new_y.T
            output = self.predict(output)
            validation_prediction[:,j:j+self.batch_size] = output
            correct_predictions = (np.argmax(output, axis=0) == np.argmax(new_y, axis=0))
            correct += np.sum(correct_predictions)  
        validation_accuracy = correct/len(x_validation)
        self.accuracy = validation_accuracy
        self.output = validation_prediction
        self.y_true = y_validation.T

        correct = 0
        for j in range(0, len(x_train), self.batch_size):
            # forward
            output = x_train[j:j+self.batch_size]
            output= output.T
            new_y = y_train[j:j+self.batch_size]
            new_y = new_y.T
            output = self.predict(output)
            train_prediction[:,j:j+self.batch_size] = output
            correct_predictions = (np.argmax(output, axis=0) == np.argmax(new_y, axis=0))
            correct += np.sum(correct_predictions)
        train_accuracy = correct/len(x_train)


        validation_loss = self.loss(validation_prediction, y_validation.T) / len(x_validation)
        train_loss = self.loss(train_prediction, y_train.T) / len(x_train)

        # validation_F1 = self.macro_f1(y_validation.T, validation_prediction)
        # train_F1 = self.macro_f1(y_train.T, train_prediction)
        validation_F1 = f1_score(np.argmax(y_validation.T, axis=0), np.argmax(validation_prediction, axis=0), average='macro')
        train_F1 = f1_score(np.argmax(y_train.T, axis=0), np.argmax(train_prediction, axis=0), average='macro')

        self.train_loss.append(train_loss)
        self.validation_loss.append(validation_loss)
        self.train_accuracy.append(train_accuracy)
        self.validation_accuracy.append(validation_accuracy)
        self.train_F1.append(train_F1)
        self.validation_F1.append(validation_F1)



    def train(self,x_validation, y_validation, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        print(y_train.shape)
        # y_train = y_train.T
        for e in range(epochs):
            error = 0
            total_accuracy = 0
            for j in range(0, len(x_train), self.batch_size):
                # forward
                output = x_train[j:j+self.batch_size]
                output= output.T
                new_y = y_train[j:j+self.batch_size]
                new_y = new_y.T
                output = self.predict(output)
                error += self.loss(output, new_y)
                # print("loss: ", err)
                grad = new_y
                # correct_predictions = (np.argmax(output, axis=0) == np.argmax(new_y, axis=0))
                # total_accuracy += np.sum(correct_predictions)  
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate, new_y)
            
            # self.calculate(e,x_validation,y_validation,x_train,y_train)
            # error /= (len(x_train)/self.batch_size)
            # average_accuracy = total_accuracy/len(x_train)
            # print('epoch %d/%d   error=%f average_accuracy=%f' % (e+1, epochs, error,average_accuracy))

            if verbose:
                print(f"{e + 1}/{epochs}, error={error/len(x_train)}")
    
    def plot(self):
        plt.plot(self.train_loss,label="train loss")
        plt.plot(self.validation_loss,label="validation loss")
        plt.legend()
        plt.show()
        plt.plot(self.train_accuracy,label="train accuracy")
        plt.plot(self.validation_accuracy,label="validation accuracy")
        plt.legend()
        plt.show()
        plt.plot(self.train_F1,label="train F1")
        plt.plot(self.validation_F1,label="validation F1")
        plt.legend()
        plt.show()

        print("validation accuracy")
        print(self.y_true.shape)
        print(self.output.shape)

        predictions = np.argmax(self.output, axis=0)
        y_true = np.argmax(self.y_true, axis=0)

        conf_matrix = confusion_matrix(y_true, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()


# write a main function
if __name__ == "__main__":
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)


    independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())
    # get the data from the folder of data
    # train_validation_dataset = 



    print("data loaded")
    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)

    X_train = np.array([sample[0].numpy().flatten() for sample in train_dataset])
    y_train = np.array([sample[1] for sample in train_dataset]) 
    X_validation = np.array([sample[0].numpy().flatten() for sample in validation_dataset])
    y_validation = np.array([sample[1] for sample in validation_dataset]) 

    X_validation = np.array([sample[0].numpy().flatten() for sample in independent_test_dataset])
    y_validation = np.array([sample[1] for sample in independent_test_dataset]) 

    X_validation_old = X_validation
    y_validation_old = y_validation


    X_validation = X_validation.reshape(X_validation.shape[0], 28 * 28, 1)
    y_validation = y_validation.reshape(y_validation.shape[0], 1)

    X_train = X_train / 255.0
    X_validation = X_validation / 255.0

    y_train = np.eye(26)[y_train-1]
    y_validation = np.eye(26)[y_validation-1]
    y_validation_old = np.eye(26)[y_validation_old-1]


    network = [
        Dense(28 * 28, 1024,XavierInitializer(),Adam()),
        Relu(),
        Dropout(0.5),
        Dense(1024, 26,XavierInitializer(),Adam()),
        Softmax()
    ]

    model = FNN(network,Loss.cross_entropy,Loss.cross_entropy_prime)
    # model.train(X_validation,y_validation,X_train, y_train, epochs=100, learning_rate=0.005)

    with open('weights.pkl', 'rb') as file:
        loaded_weights = pickle.load(file)

    with open('bias.pkl', 'rb') as file:
        loaded_bias = pickle.load(file)

    # Set the loaded weights to the model
    i=0
    for layer in network:
        layer.set_weights(loaded_weights[f'layer_{i}_weights'])
        layer.set_bias(loaded_bias[f'layer_{i}_bias'])
        i += 1


    # fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    # lr=0.01
    
    # model = FNN(network,Loss.cross_entropy,Loss.cross_entropy_prime)
    # model.train(X_validation,y_validation,X_train, y_train, epochs=30, learning_rate=lr)

    # fig, axs = plt.subplots(2,2, figsize=(16, 12))

    # axs[0,0].plot(model.train_loss, label=f'Train Loss LR={lr}')
    # axs[0,0].plot(model.validation_loss, label=f'Validation Loss LR={lr}')
    # axs[0,0].set_title(f'Loss LR={lr}')
    # axs[0,0].set_xlabel('Epochs')
    # axs[0,0].set_ylabel('Loss')
    # axs[0,0].legend()

    # # Plot accuracy for each learning rate
    # axs[0,1].plot(model.train_accuracy, label=f'Train Accuracy LR={lr}')
    # axs[0,1].plot(model.validation_accuracy, label=f'Validation Accuracy LR={lr}')
    # axs[0,1].set_title(f'Accuracy LR={lr}')
    # axs[0,1].set_xlabel('Epochs')
    # axs[0,1].set_ylabel('Accuracy')
    # axs[0,1].legend()

    # axs[1,0].plot(model.train_F1, label=f'Train F1 score LR={lr}')
    # axs[1,0].plot(model.validation_F1, label=f'Validation F1 score LR={lr}')
    # axs[1,0].set_title(f'F1 score LR={lr}')
    # axs[1,0].set_xlabel('Epochs')
    # axs[1,0].set_ylabel('F1 score')
    # axs[1,0].legend()

    # predictions = np.argmax(model.output, axis=0)
    # y_true = np.argmax(model.y_true, axis=0)

    # conf_matrix = confusion_matrix(y_true, predictions)

    # # axs[i, 3].figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=['PN', 'PP'],
    #             yticklabels=['TN', 'TP'])
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.title('Confusion Matrix')
    

    # plt.show()

    # print("train loss", model.train_loss[-1])
    # print("validation loss", model.validation_loss[-1])
    # print("train accuracy", model.train_accuracy[-1])
    # print("validation accuracy", model.validation_accuracy[-1])
    # print("train F1", model.train_F1[-1])
    # print("validation F1", model.validation_F1[-1])



    correct = 0
    for x, y in zip(X_validation, y_validation):
        output = model.predict(x)
        if np.argmax(output) == np.argmax(y):
            correct += 1


    # print(len(X_validation))
    print('Accuracy:', correct / len(X_validation) * 100)

    #  calculate loss
    loss = 0
    for x,y in zip(X_validation,y_validation):
        output = model.predict(x)
        loss += model.loss(output,y)
    print("loss: ", loss/len(X_validation))


    # calculate F1 score
    validation_prediction = np.zeros((26,len(X_validation_old)))
    batch_size = 1024
    for j in range(0, len(X_validation_old), 1024):
            # forward
            output = X_validation_old[j:j+batch_size]
            output= output.T
            new_y = y_validation_old[j:j+batch_size]
            new_y = new_y.T
            output = model.predict(output)
            validation_prediction[:,j:j+batch_size] = output
    validation_F1 = f1_score(np.argmax(y_validation_old.T, axis=0), np.argmax(validation_prediction, axis=0), average='macro')
    print("F1 score: ", validation_F1)