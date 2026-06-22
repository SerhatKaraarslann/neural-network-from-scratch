import numpy as np

# Importieren der Layer, Aktivierungsfunktionen, Verlustfunktion und Optimizer
from layer import Layer_Dense
from activations import Activation_ReLU, Activation_Softmax
from loss import Loss_CategoricalCrossentropy
from optimizer import Optimizer_SGD

class NeuralNetwork:
    """
    Die Hauptklasse unseres neuronalen Netzwerks.
     
    Hier werden die Layer, Aktivierungsfunktionen, Verlustfunktion und Optimizer definiert und miteinander verbunden.
    Wir steuern den Trainingsprozess, indem wir die Vorwärts- und Rückwärtsdurchläufe durch das Netzwerk durchführen und die Gewichte aktualisieren.
    Methods:
    __init__: Konstruktor, der die Layer, Aktivierungsfunktionen, Verlustfunktion und Optimizer initialisiert.
    train_step: Methode, die einen einzelnen Trainingsschritt durchführt, einschließlich Vorwärtsdurchlauf, Verlustberechnung, Rückwärtsdurchlauf und Gewichtsaktualisierung.
    """

    def __init__(self):
        """
        Konstruktor für die NeuralNetwork Klasse. Hier werden die Layer, Aktivierungsfunktionen, Verlustfunktion und Optimizer initialisiert.
        Für unser Ziel, MNIST zu klassifizieren, haben wir einen Input Layer mit 784 Eingängen (28x28 Pixel), einen versteckten Layer mit 128 Neuronen und ReLU-Aktivierung, und einen Output Layer mit 10 Neuronen (für die 10 Klassen) und Softmax-Aktivierung. 
        Wir verwenden die Categorical Crossentropy als Verlustfunktion und Stochastic Gradient Descent (SGD) als Optimizer.
        """

        # Schichten des Netzwerks definieren
        self.layer1 = Layer_Dense(784, 128) # Input Layer mit 784 Eingängen und verstecktem Layer mit 128 Neuronen
        self.activation1 = Activation_ReLU() # ReLU-Aktivierungsfunktion für den versteckten Layer
    
        self.layer2 = Layer_Dense(128, 10) # Output Layer mit 128 Eingängen (vom versteckten Layer) und 10 Neuronen (für die 10 Klassen)
        self.activation2 = Activation_Softmax() # Softmax-Aktivierungsfunktion für den Output Layer


        # Verlustfunktion und Optimizer definieren
        self.loss_function = Loss_CategoricalCrossentropy() # Verlustfunktion: Categorical Crossentropy
        self.optimizer = Optimizer_SGD(learning_rate=0.01) # Optimizer: Stochastic Gradient Descent mit einer Lernrate von 0.01

    def train_step(self, X, y):
        """
        Ein kompletter Trainingsschritt durch das Netzwerk. Hier führen wir den Vorwärtsdurchlauf, die Verlustberechnung, den Rückwärtsdurchlauf und die Gewichtsaktualisierung durch.
        :param X: Eingabedaten, Bilder
        :param y: Zielwerte (Labels)
        :return: Den Verlust und die Vorhersagen für diesen Trainingsschritt
        """
        # Vorwärtsdurchlauf
        self.layer1.forward(X) # Vorwärtsdurchlauf durch den ersten Layer
        self.activation1.forward(self.layer1.output) # Vorwärtsdurchlauf durch die ReLU-Aktivierung
        self.layer2.forward(self.activation1.output) # Vorwärtsdurchlauf durch den zweiten Layer
        self.activation2.forward(self.layer2.output) # Vorwärtsdurchlauf durch die Softmax-Aktivierung

        # Verlust berechnen
        loss = self.loss_function.calculate(self.activation2.output, y) # Berechnung des Verlusts basierend auf den Vorhersagen und den Zielwerten

        # Rückwärtsdurchlauf
        self.loss_function.backward(self.activation2.output, y) # Rückwärtsdurchlauf durch die Verlustfunktion
       
        # Der Fehler fließt weiter rückwärts durch das Netzwerk
        self.layer2.backward(self.loss_function.dinputs) # Rückwärtsdurchlauf durch den zweiten Layer
        self.activation1.backward(self.layer2.dinputs) # Rückwärtsdurchlauf durch die ReLU-Aktivierung
        self.layer1.backward(self.activation1.dinputs) # Rückwärtsdurchlauf durch den ersten Layer

        # Gewichte aktualisieren    
        self.optimizer.update_params(self.layer1) # Aktualisierung der Gewichte und Biases des ersten Layers
        self.optimizer.update_params(self.layer2) # Aktualisierung der Gewichte und Biases des zweiten Layers
    

        # Vorhersagen zurückgeben
        predictions = np.argmax(self.activation2.output, axis=1) # Vorhersagen basierend auf der Ausgabe des Softmax-Layers (die Klasse mit der höchsten Wahrscheinlichkeit)

        return loss, predictions
    
    def evaluate(self, X, y):
        """
        Testet das Netzwerk auf dem gegebenen Testdatensatz und berechnet die Genauigkeit, ohne zu trainieren (kein Rückwärtsdurchlauf oder Gewichtsaktualisierung).
        :param X: Eingabedaten, Bilder
        :param y: Zielwerte (Labels)
        :return: Die Genauigkeit des Netzwerks auf dem Testdatensatz
        """
        # Vorwärtsdurchlauf (ohne Rückwärtsdurchlauf und Gewichtsaktualisierung)
        self.layer1.forward(X) # Vorwärtsdurchlauf durch den ersten Layer
        self.activation1.forward(self.layer1.output) # Vorwärtsdurchlauf durch die ReLU-Aktivierung
        self.layer2.forward(self.activation1.output) # Vorwärtsdurchlauf durch den zweiten Layer
        self.activation2.forward(self.layer2.output) # Vorwärtsdurchlauf durch die Softmax-Aktivierung
        
        # Vorhersagen und Loss zurückgeben
        predictions = np.argmax(self.activation2.output, axis=1) # Vorhersagen basierend auf der Ausgabe des Softmax-Layers (die Klasse mit der höchsten Wahrscheinlichkeit)
        loss = self.loss_function.calculate(self.activation2.output, y) # Berechnung des Verlusts basierend auf den Vorhersagen und den Zielwerten

        # Genauigkeit berechnen
        accuracy = np.mean(predictions == y) # Berechnung der Genauigkeit als Anteil der korrekten Vorhersagen

        return loss , accuracy