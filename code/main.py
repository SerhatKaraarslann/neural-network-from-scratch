import numpy as np
from netzwerk import NeuralNetwork

from tensorflow.keras.datasets import mnist # Importieren des MNIST-Datensatzes

class Trainer:
    """
    Die Trainer-Klasse steuert den Trainingsprozess unseres neuronalen Netzwerks.
    Hier laden wir die Daten, bereiten sie vor und führen den Trainingsprozess durch.
    Methods:
    __init__: Konstruktor, der das neuronale Netzwerk initialisiert.
    train: Methode, die den Trainingsprozess über mehrere Epochen steuert.
    test: Methode, die die Leistung des Netzwerks auf den Testdaten bewertet.
    """

    def __init__(self, network):
        """
        Konstruktor für die Trainer-Klasse. Hier wird das neuronale Netzwerk initialisiert.
        """
        self.network = network # Initialisierung des neuronalen Netzwerks

    def train(self, x_train, y_train, epochs=10 , batch_size=128):
        """
        Methode, die den Trainingsprozess über mehrere Epochen steuert.
        :param X_train: Trainingsdaten, Bilder
        :param y_train: Zielwerte (Labels) für die Trainingsdaten
        :param epochs: Anzahl der Epochen, die das Netzwerk trainieren soll
        :param batch_size: Größe der Mini-Batches für das Training
        """
        steps_per_epoch = len(x_train) // batch_size # Berechnung der Anzahl der Schritte pro Epoche basierend auf der Batch-Größe

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}') # Ausgabe der aktuellen Epoche

            # Wir schneiden eine Epoche in Mini-Batches und führen für jeden Batch einen Trainingsschritt durch
            for step in range(steps_per_epoch):
                
                start = step * batch_size # Startindex für den aktuellen Batch
                end = start + batch_size # Endindex für den aktuellen Batch

                batch_x = x_train[start:end] # Auswahl der Bilder für den aktuellen Batch
                batch_y = y_train[start:end] # Auswahl der Labels für den aktuellen Batch

                # Durchführung eines Trainingsschritts mit dem aktuellen Batch
                loss, predictions = self.network.train_step(batch_x, batch_y) # Trainingsschritt

                # Genauigkeit berechnen
                accuracy = np.mean(predictions == batch_y) # Berechnung der Genauigkeit für den aktuellen Batch

                if step % 100 == 0: # Ausgabe des Fortschritts alle 100 Schritte
                    print(f'Step {step}/{steps_per_epoch} - Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%') # Ausgabe von Verlust und Genauigkeit für den aktuellen Batch

    def test(self, x_test, y_test):
        """
        Methode, die die Leistung des Netzwerks auf den Testdaten bewertet.
        Hier testen wir das Netzwerk auf den Testdaten, Ohne zu tranieren. 
        Das Flißband läuft nur vorwärts,  keine Gewichtsaktualisierung findet statt. Wir berechnen die Vorhersagen und die Genauigkeit auf den Testdaten.
        :param x_test: Testdaten, Bilder
        :param y_test: Zielwerte (Labels) für die Testdaten
        """
        # Vorwärtsdurchlauf durch das Netzwerk mit den Testdaten, um die Vorhersagen zu erhalten
        loss, accuracy = self.network.evaluate(x_test, y_test) # Bewertung des Modells auf den Testdaten, Berechnung von Verlust und Genauigkeit

        print(f"Finale Genauigkeit auf den Testdaten: {accuracy * 100:.2f}%") # Ausgabe der finalen Genauigkeit auf den Testdaten
        print(f"Finaler Verlust auf den Testdaten: {loss:.4f}") #

        return accuracy, loss # Rückgabe der Genauigkeit und des Verlusts auf den Testdaten
    
def preprocess_data(x):
    """
    Hilfsfunktion zur Vorverarbeitung der Daten. Hier normalisieren wir die Pixelwerte der Bilder, damit sie im Bereich von 0 bis 1 liegen.
    :param x: Eingabedaten, Bilder
    :return: Vorverarbeitete Daten, normalisierte Bilder
    """
    x = x.reshape(x.shape[0], -1) # Umwandlung der Bilder von 28x28 in einen Vektor der Länge 784
    x = x.astype('float32') / 255.0 # Normalisierung der Pixelwerte auf den Bereich [0, 1]
    return x

# Hauptfunktion, die den gesamten Trainings- und Testprozess steuert
if __name__ == "__main__":

    # Laden des MNIST-Datensatzes
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # Laden der Trainings- und Testdaten

    # Vorverarbeitung der Daten
    x_train = preprocess_data(x_train) # Vorverarbeitung der Trainingsdaten
    x_test = preprocess_data(x_test) # Vorverarbeitung der Testdaten

   # Netzwerk initialisieren und Trainer erstellen
    network = NeuralNetwork() # Initialisierung des neuronalen Netzwerks
    trainer = Trainer(network) # Initialisierung des Trainers
    
    # Training des Netzwerks
    trainer.train(x_train, y_train, epochs=10, batch_size=128) # Start des Trainingsprozesses mit den Trainingsdaten, 10 Epochen und einer Batch-Größe

    # Testen des Netzwerks
    trainer.test(x_test, y_test) # Start des Testprozesses mit den Testdaten