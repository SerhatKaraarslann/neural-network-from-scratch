### Tag 4: Verlustfunktion und Backpropagation - Das Lernen beginnt
Willkommen zu Tag 4! In den Tag 3 haben wir unserem Netzwerk ein Motor gegeben, damit es komplexe Muster erkennen kann. Unser Fließband ist jetzt komplett. Daten gehen rein, fließen durch Dense Layers, werden durch ReLU nicht-linear verarbeitet und am Ende in der Softmax Layer in Wahrscheinlichkeiten umgewandelt.

Das Problem ist, dass unser Netzwerk rät völlig zufällig. Es spuckt am Anfang einfach Müll aus, weil die Gewichte zufällig sind. Wenn ich dem Netzwerk ein Bild von einer 8 zeige, rät es völlig blind. Es sagt mir vllt. "Das ist eine 3 mit 40% Wahrscheinlichkeit, eine 7 mit 30% Wahrscheinlichkeit, ...". Das ist natürlich falsch. 

Hier brauche ich einen extrem strengen Lehrer, der die Vorhersagen des Netzwerks bewertet, eine Note vergibt und sagt: "Das war schlecht, du musst deine Gewichte anpassen, damit du das nächste Mal besser rätst!". In der KI-Welt nennt man diesen Lehrer "Verlustfunktion" und die Methode, mit der das Netzwerk aus seinen Fehlern lernt, "Backpropagation". 

### Die Loss Basisklasse 
Bevor ich mich auf die spezielle Mathematik gestürtz habe, habe ich eine allgemeine Basisklasse für Verlustfunktion geschrieben. Warum? Weil es verschiedene Arten von Verlustfunktionen gibt, je nachdem, was für ein Problem wir lösen wollen. Aber egal welche Verlustfunktion wir später verwenden, wollen wir am Ende immer eine Zahl herausbekommen, die uns sagt, wie schlecht unsere Vorhersage war. Diese Zahl nennen wir "Loss" oder "Verlust". Je höher der Verlust, desto schlechter war die Vorhersage. In meiner Basisklasse habe ich eine Methode namens calculate(), die die Vorhersagen des Netzwerks und die wahren Labels (die richtigen Antworten) entgegennimmt und den durchschnittlichen Verlust über alle Beispiele berechnet. Das ist die Zahl, die wir am Ende des Tages sehen wollen, um zu wissen, wie schlecht unser Netzwerk gerade ist.

### Categorical Cross-Entropy: Unsere Verlustfunktion für Klassifikation
Da mein Netzwerk MNIST-Ziffern in Kategorien einteilen soll, erbt meine eingentliche Verlustfunktion von der Basisklasse und nennt sich Categorical Cross-Entropy.

Diese Funktiion schaut sich nur die Wahrscheinlichkeiten an, die das Netzwerk für die richtige Klasse ausgegeben hat und alles andere ignoriert. 
Die richtige Antworten können in zwei Formen vorliegen: entweder als 1D-Array (z.B. [3, 7, 1]) oder als One-Hot-Vektor, 2D Array (z.B. [0, 0, 0, 1], [1, 0, 0, 0]).
Falls es als einfaches 1D-Array vorliegt, pickt die Funktion diese genau die Wahrscheinlichkeit aus der Vorhersage heraus, die der richtigen Klasse entspricht. Wenn es als One-Hot-Vektor vorliegt, multipliziert die Funktion die Vorhersage mit dem One-Hot-Vektor und summiert dann über alle Klassen, um die Wahrscheinlichkeit der richtigen Klasse zu erhalten.

Sobald ich die Wahrscheinlichkeit der richtigen Klasse habe, berechnet die Funktion den Verlust mit der Formel:
Loss = -log(probability_of_correct_class)
- Je höher die Wahrscheinlichkeit der richtigen Klasse, desto niedriger der Verlust. Wenn das Netzwerk also eine hohe Wahrscheinlichkeit für die richtige Klasse ausspuckt, ist der Verlust niedrig, was gut ist. Wenn das Netzwerk eine niedrige Wahrscheinlichkeit für die richtige Klasse ausspuckt, ist der Verlust hoch, was schlecht ist.

- Szenario A: Das Netzwerk war schlau. Das Netz hat 0.95 für die richtige Antwort ausgegeben. Der Verlust wäre dann -log(0.95) = 0.05, was sehr gut ist.
- Szenario B: Das Netzwerk war dumm. Das Netz hat 0.01 für die richtige Antwort ausgegeben. Der Verlust wäre dann -log(0.01) = 4.6, was sehr schlecht ist.

### Geheimtrick in der Code - Stabilität bei Categorical Cross-Entropy
Beim Programmieren bin ich auf ein gefährliches Problem gestoßen. Was passiert, wenn das Netzwerk völlig daneben liegt und der richtigen Antwort eine Wahrscheinlichkeit von exakt 0.0 zuordnet?
Dann würde die Funktion versuchen, -log(0.0) zu berechnen. Das ist mathematisch gesehen unendlich! Python würde versuchen, diese Zahl zu berechnen und entweder sofort abstürzen oder mit unendlichen Werten arbeiten, was das gesamte Training zerstören würde.

Um dieses Problem zu umgehen, habe ich einen kleinen Trick eingebaut: Ich nutze die Funktion np.clip() von NumPy, um die Wahrscheinlichkeiten zu begrenzen. Ich setze die absolut kleinsten Werte heimlich auf 1e-7 (also 0.0000001) und die oberen Werte auf 1 - 1e-7 (also 0.9999999).
Dadurch stelle ich sicher, dass die Funktion niemals -log(0.0) berechnet, sondern stattdessen -log(1e-7). Das ergibt eine sehr große Zahl als Strafe, aber sie kann immer noch fehlerfrei berechnet werden. Mathematisch ändert das am Endergebnis gar nichts, aber es schützt mein Programm vor dem Absturz!

### Fazit für heute
Heute haben wir den Lehrer für unser Netzwerk gebaut: die Verlustfunktion. Sie bewertet die Vorhersagen des Netzwerks und gibt uns eine Note.

Jetzt haben wir die Vorwärtsausbreitung komplett. Das Netzwerk kann Daten durch die Dense Layers schicken, sie mit ReLU in den Hidden Layers verarbeiten, am Ende mit Softmax eine Wahrscheinlichkeitsverteilung über die möglichen Zahlen bekommen und die Verlustfunktion kann bewerten, wie schlecht diese Vorhersage war. 

Ein schlectes Zeugnis zu bekommen, reicht aber nicht aus. Das Netzwerk muss aus seinen Fehlern lernen! Es muss jetzt das gesamte Fließband rückwärts durchlaufen und herausfinden, welches einzelne Neuron und welches Gewicht eigentlich an diesem schlechten Zeugnis schuld war. 

Genau das ist die gefürchtete Backpropagation (Rückwärtsausbreitung). Sie ist der Teil, in dem das Netzwerk berechnet, wie sehr es sich geirrt hat und seine Gewichte anpasst. Aber bevor das Netz aus seinen Fehlern lernen kann, müssen wir erstmal den Vorwärtsdurchlauf komplett fertig bauen und den Fehler überhaupt messen können! Deshalb bleibt die Backpropagation heute noch leer. 