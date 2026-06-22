import numpy as np

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Nimmt die Gradienten der Gewichte und Biases und aktualisiert sie in Richtung des negativen Gradienten, um den Fehler zu minimieren.
    Formel: w = w - learning_rate * dw
           b = b - learning_rate * db
    Anwendung: Häufig in der Trainingsphase von neuronalen Netzwerken verwendet, um die Gewichte und Biases zu aktualisieren.
    Attrubutes:
    learning_rate: Die Lernrate bestimmt, wie groß die Schritte sind, die wir bei der Aktualisierung der Gewichte und Biases machen.
    Methods:
    __init__: Konstruktor, um die Lernrate zu initialisieren.
    update_params: Methode, die die Gewichte und Biases basierend auf den berechneten Gradienten aktualisiert.
    """

    def __init__(self,learning_rate=0.01):
        """
        Konstruktor für den SGD Optimizer.
        :param learning_rate : Die Lernrate. Standart ist meistens 1.0, 0.1 oder 0.01.Sie bestimmt, wie groß die Schritte sind, die wir bei der Aktualisierung der Gewichte und Biases machen.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Aktualisiert die Gewichte und Biases eines Layers basierend auf den berechneten Gradienten.
        :param layer: Der Layer, dessen Gewichte und Biases aktualisiert werden sollen. Der Layer muss die Attribute weights, biases, dw und db haben.
        """
        # Aktualisieren der Gewichte: w = w - learning_rate * dw
        layer.weights += -self.learning_rate * layer.dweights

        # Aktualisieren der Biases: b = b - learning_rate * db
        layer.biases += -self.learning_rate * layer.dbiases