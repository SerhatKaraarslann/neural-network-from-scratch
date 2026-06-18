import numpy as np

class Loss:
    """
    Loss - Basisklasse für Verlustfunktionen.
    Egal welche Verlustfunction wir verwenden, 
    sie sollten von dieser Klasse erben, damit wir die gleiche Schnittstelle für die Berechnung des Verlusts haben.
    
    Methods:
    calculate: Methode zur Berechnung des Verlusts."""

    def calculate(self, output, y):
        """
        Berechnet den Verlust.
        :param output : Die Vorhersagen des Modells aus der Softmax Schicht.
        :param y: Die tatsächlichen, echten Antorten (Labels).
        :return: Der berechnete Verlustwert.
        """

        # Rufe die forward Methode der spezifischen Verlustfunktion auf, um den Verlust für jedes Sample zu berechnen.
        sample_losses = self.forward(output, y)

        # Berechne den Durchschnittsverlust über alle Samples in der Batch.
        data_loss = np.mean(sample_losses)

        return data_loss
    

class Loss_CategoricalCrossentropy(Loss):
    """
    Die kategorische Kreuzentropie ist eine häufig verwendete Verlustfunktion für Klassifikationsprobleme, insbesondere bei der Mehrklassenklassifikation.
    Formel: L = -sum(y_true * log(y_pred)) / N
    Anwendung: Häufig in Kombination mit der Softmax-Aktivierungsfunktion in der Output Layer verwendet, um die Differenz zwischen den vorhergesagten Wahrscheinlichkeiten und den tatsächlichen Klassenlabels zu messen.
    Methods:
    forward: Methode für die Vorwärtsausbreitung, die die Ausgabe des Layers berechnet.
    backward: Methode für die Rückwärtsausbreitung, die die Gradienten für die Gewichte und Biases berechnet.
     """

    def forward(self, y_predicted, y_true):
        """
        Der Vorwärtsdurchlauf der kategorischen Kreuzentropie-Verlustfunktion.
        :param y_predicted: Die Vorhersagen des Modells aus der Softmax-Schicht.
        :param y_true: Die tatsächlichen, echten Antorten (Labels).
        :return: Der berechnete Verlustwert für jedes Sample.
        """

        # Anzahl der Samples in der Batch, wie viele Vorhersagen wir haben.
        samples = len(y_predicted)

        # Trick für die numerische Stabilität: 
        # log(0) ist undefiniert und führt zu einem Fehler.
        # Wir fügen eine kleine Zahl (1e-7 = 0.0000001) zu den Vorhersagen hinzu, um zu verhindern, dass log(0) berechnet wird, was zu einem Fehler führen würde.
        # Außerdem stellen wir sicher, dass die Vorhersagen nicht größer als 1.0 werden, da log(>1) auch zu einem Fehler führen könnte.
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            # wenn y_true, die Labels, eindimensional sind sowie z,B. [0, 1, 2, 3, ...], dann holen wir die genaue Wahrscheinlichekeit der echten Klasse für jedes Sample.
            correct_confidences = y_predicted_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            # wenn y_true, die Labels, zweidimensional sind sowie z.B. [[0, 1, 0], [0, 0, 1], ...], dann berechnen wir die Wahrscheinlichkeiten der echten Klassen durch Elementweise Multiplikation und Summierung über die Klassenachse.
            correct_confidences = np.sum(y_predicted_clipped * y_true, axis=1)

        
        # Eigentliche Berechnung der Cross-Entropy-Verlusts: 
        # Wir nehmen den negativen Logarithmus der Wahrscheinlichkeiten der echten Klassen, um den Verlust zu berechnen.
        # Je höher die Wahrscheinlichkeit der echten Klasse, desto niedriger der Verlust. 
        # Wenn die Vorhersage gut ist (Wahrscheinlichkeit nahe 1), ist der Verlust nahe 0.
        # Wenn die Vorhersage schlecht ist (Wahrscheinlichkeit nahe 0), ist der Verlust hoch.
        negative_log = -np.log(correct_confidences)

        return negative_log
    
    def backward(self, dvalues, y_true):
        """
        Rückwärtsdurchlauf der kategorischen Kreuzentropie-Verlustfunktion, für Loss und Softmax kombiniert.
        Dies ist der Startschuss für die Backpropagation.
        Hier berechnen wir die Gradienten für die Eingänge basierend auf den Gradienten der Ausgabe (dvalues) und den tatsächlichen Labels (y_true).
        :param dvalues: Die Gradienten(Fehlerwerte) der Ausgabe, die von der nächsten Schicht zurückgegeben werden.
        :param y_true: Die tatsächlichen, echten Antorten (Labels).
        """

        # Anzahl der Samples in der Batch, wie viele Vorhersagen wir haben.
        samples = len(dvalues)

        # wenn die Labels als One-Hot-Vektoren(2D) vorliegen, 
        # wandeln wir sie in einfache Klassenindizes(1D) um, damit wir die Gradienten korrekt berechnen können.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Wir machen eine Kopie der Gradienten der Ausgabe, damit wir sie modifizieren können, ohne die Originalwerte zu verändern.
        self.dinputs = dvalues.copy()

        # Softmax und Cross-Entropy zusammen:
        # Wenn wir die Softmax-Aktivierungsfunktion in der Output Layer verwenden und die kategorische Kreuzentropie als Verlustfunktion, 
        # können wir die Gradienten für die Eingänge direkt berechnen, indem wir die Vorhersagen (dvalues) von den tatsächlichen Labels (y_true) subtrahieren.
        # Dies ist eine mathematische Vereinfachung, die sich aus der Kombination von Softmax und Kreuzentropie ergibt.
        self.dinputs[range(samples), y_true] -= 1

        # Wir teilen die Gradienten durch die Anzahl der Samples, um den Durchschnittsgradienten zu erhalten.
        self.dinputs /= samples