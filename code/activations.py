import numpy as np

class Activation_ReLU:
    """
    ReLU(Rectified Linear Unit) Aktivierungsfunktion
    Gibt den Input zurück, wenn er positiv ist, andernfalls 0.
    Formel: f(x) = max(0, x)
    Anwendung: Häufig in versteckten Schichten von neuronalen Netzwerken verwendet.

    Methods:
    forward: Methode für die Vorwärtsausbreitung, die die Ausgabe des Layers berechnet.
    backward: Methode für die Rückwärtsausbreitung, die die Gradienten für die Gewichte und Biases berechnet.
    """
    def forward(self, inputs) -> np.ndarray:
        """
        Vorwärtsdurchlauf der ReLU-Aktivierungsfunktion.
        :param inputs: Eingabedaten (z.B. von einer vorherigen Schicht)
        :return: Aktivierte Ausgabe nach Anwendung der ReLU-Funktion.
        """

        # Wir speichern die Inputs wieder für die Backpropagation
        self.input = inputs

        # np.max vergelicht jeden Wert in inputs mit 0.
        # Wenn der Wert größer als 0 ist, wird er beibehalten. Ist er <= 0, wird er zu 0.
        self.output = np.max(0, inputs) # Elementweise Anwendung der ReLU-Funktion

        return self.output
    
    def backward(self):
        pass

class Activation_Softmax:
    """
    Softmax Aktibierungsfunktion.
    Gibt die Wahrscheinlichkeitsverteilung über mehrere Klassen zurück.
    Formel: f(x_i) = exp(x_i) / sum(exp(x_j)) für alle j
    Anwendung: Häufig in der Output Layer verwendet, um die rohen Werte (Logits) in Wahrscheinlichkeiten (0.0 - 1.0) umzuwandeln.
    
    Methods:
    forward: Methode für die Vorwärtsausbreitung, die die Ausgabe des Layers berechnet.
    backward: Methode für die Rückwärtsausbreitung, die die Gradienten für die Gewichte und Biases berechnet.
    """

    def forward(self, inputs) -> np.ndarray:
        """
        Vorwärtsdurchlauf der Softmax-Aktivierungsfunktion.
        :param inputs: Eingabedaten (z.B. von einer vorherigen Schicht)
        :return: Aktivierte Ausgabe nach Anwendung der Softmax-Funktion.
        """

        # Wir speichern die Inputs wieder für die Backpropagation
        self.input = inputs

        # Um numerische Stabilität zu gewährleisten, subtrahieren wir den größten Wert in jeder Zeile von allen Werten in der Zeile.
        # Wir ziehen den größten Wert in jeder Zeile von allen Werten ab.
        # Das ändert mathematisch nichts am finalen Wahrscheinlichkeits-Ergebnis,
        # verhindert aber, dass np.exp() bei riesigen Zahlen abstürzt.
        # keepdims=True sorgt dafür, dass die Form der Matrix erhalten bleibt.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Berechnung der Softmax-Ausgabe durch Normalisierung der Exponentialwerte
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output
    
    def backward(self):
        pass