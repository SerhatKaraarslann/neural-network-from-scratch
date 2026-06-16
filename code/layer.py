import numpy as np # für matrix erstellung und operationen

class Layer_Dense:
    """Dense Layer - also fully connected layer klasse.
    
    Attributes:
    number_of_inputs: Anzahl der Eingänge
    number_of_neurons: Anzahl der Neuronen
    weights: Gewichte der Verbindungen zwischen Eingängen und Neuronen
    biases: Biases der Neuronen

    Methods:
    __init__: Konstruktor, der die Anzahl der Eingänge und Neuronen initialisiert und die Gewichte und Biases zufällig generiert.
    forward: Methode für die Vorwärtsausbreitung, die die Ausgabe des Layers berechnet.
    backward: Methode für die Rückwärtsausbreitung, die die Gradienten für die Gewichte und Biases berechnet.
    """
    def __init__(self, number_of_inputs : int, number_of_neurons : int):
        """Konstruktor für die Dense Layer Klasse. Hier kann man die Anzahl der Eingänge und Neuronen angeben, und die Gewichte und Biases werden zufällig generiert.
        :param number_of_inputs: Anzahl der Eingänge
        :param number_of_neurons: Anzahl der Neuronen
    """
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.weights = 0.01 * np.random.randn(number_of_inputs, number_of_neurons) # Matrix der Gewichte, zufällig initialisiert. Größe: Anzahl der Eingänge x Anzahl der Neuronen
        self.biases = np.zeros((1,number_of_neurons)) # Matrix der Biases, initialisiert mit Nullen, Größe: 1 x Anzahl der Neuronen

    def forward(self, inputs) -> np.ndarray:
        """Methode für die Vorwärtsausbreitung. Hier wird die Ausgabe des Layers berechnet, indem die Eingänge mit den Gewichten multipliziert und die Biases addiert werden.
        :param inputs: Eingabedaten (z.B. von einer vorherigen Schicht)
        :return: Ausgabe des Layers nach der Berechnung
        """
        self.inputs = inputs # Speichern der Eingänge für die Rückwärtsausbreitung
        self.output = np.dot(inputs, self.weights) + self.biases # Berechnung der Ausgabe
        
        return self.output
        
    
    def backward(self):
        pass