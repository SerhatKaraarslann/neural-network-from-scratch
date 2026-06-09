### Tag 2 : Meine erste Schicht im Code – Warum einzelne Neuronen eine blöde Idee sind
Willkommen zu Tag 2! Im Tag 1 haben wir uns den Bauplan aufgemalt und verstanden, was ein Neuronales Netzwerk überhaupt ist. Heute machen wir uns die Hände schmutzig und bauen unseren allerersten Lego-Stein in Python: Die Schicht (Layer).

Aber bevor ich losgetippt habe, stand ich vor einem logischen Problem.
# Der Anfänger-Fehler: Warum ich keine Neuron-Klasse baue
Mein erster Gedanke war: "Okay, ein Netzwerk besteht aus Neuronen. Also schreibe ich eine Klasse namens Neuron und packe hunderte davon in eine Liste." Klingt logisch, wäre aber eine Katastrophe geworden! Wenn ich für jedes Neuron einzeln die Gewichte anpasse und die Ausgaben berechne, müsste ich riesige for-Schleifen in Python schreiben. Und Python ist bei Schleifen leider extrem langsam. Das Training würde ewig dauern.

Deshalb habe ich mich für den clevereren Weg entschieden: Ich baue direkt die ganze Schicht (Layer). Indem ich alle Eingänge, Gewichte und Biases in große Zahlentabellen (Matrizen) packe, kann der Computer alle Neuronen einer Schicht auf einen Schlag berechnen. Das ist nicht nur viel schneller, sondern auch viel eleganter. Statt hunderte von Neuronen einzeln zu berechnen, kann ich mit einer einzigen Zeile Code die gesamte Schicht durchrechnen lassen. Das ist der Schlüssel zur Effizienz in Neuronalen Netzwerken.

### Die "Dense Layer": Unser Allrounder
Die Schicht, die ich heute geschrieben habe, nennt sich Dense Layer (oder "Fully Connected Layer"). "Dicht", weil hier wirklich jeder Eingang mit absolut jedem Neuron verbunden ist.

Die Mathematik dahinter ist eigentlich total simpel:
Ausgabe = (Eingänge * Gewichte) + Biases

### Der Bauplan: Mein Python-Code
In meiner Datei habe ich eine Klasse namens Layer_Dense angelegt. Sie hat eigentlich drei Hauptaufgaben, aber eine davon hebe ich mir für später auf:

1. Die Vorbereitung (Der Konstruktor __init__)
Wenn ich die Schicht erschaffe, muss ich ihr nur sagen: Wie viele Eingänge kommen rein? Und wie viele Neuronen willst du haben? * Die Gewichte generiere ich komplett zufällig (damit nicht alle Neuronen das Gleiche lernen). Ich multipliziere sie mit 0.01, damit die Startwerte sehr klein sind – sonst schießen die Zahlen später durch die Decke und das Training crasht.

    Die Biases setze ich zum Start einfach alle auf Nullen. Das ist eine gängige Praxis, weil die Gewichte ja schon zufällig sind. Es gibt keinen Grund, die Biases von Anfang an zu stören. 

2. Die Arbeit (Der Vorwärtsdurchlauf forward)
Hier fließen die Daten durch die Schicht. Die Methode nimmt die Eingabedaten und rechnet unsere Formel aus.
Ein kleines Geheimnis: Ich speichere die inputs extra in der Klasse ab (self.inputs = inputs). Warum? Weil ich diese Originaldaten später für die Backpropagation (das Lernen aus Fehlern) unbedingt wieder brauche!

3. Das Lernen (Der Rückwärtsdurchlauf backward)
Wenn ihr in meinen Code schaut, seht ihr da noch eine backward-Methode, in der aktuell nur pass steht. Warum ist die leer?
Das ist der Platzhalter für die Backpropagation – also den Teil, in dem das Netzwerk berechnet, wie sehr es sich geirrt hat und seine Gewichte anpasst. Aber bevor das Netz aus seinen Fehlern lernen kann, müssen wir erstmal den Vorwärtsdurchlauf komplett fertig bauen und den Fehler überhaupt messen können! Deshalb bleibt diese Methode heute noch leer. Das dicke Mathe-Brett bohren wir erst in ein paar Tagen.

## np.dot: Meine absolute Geheimwaffe
Wie rechne ich das alles jetzt ohne langsame for-Schleifen aus? Die Antwort heißt NumPy.

NumPy ist die absolute Standard-Bibliothek für Mathe in Python. Statt jedes Neuron einzeln zu multiplizieren, nutze ich die Funktion np.dot() (das sogenannte Skalarprodukt). Diese eine Funktion multipliziert meine kompletten Eingangs-Matrizen mit meinen Gewichts-Matrizen und addiert die Biases in einem einzigen, extrem optimierten Schritt. Das macht den Code nicht nur tausendmal schneller, sondern er liest sich auch viel sauberer.

### Fazit für heute
Mein erster Baustein steht! Ich habe eine Schicht, die blitzschnell Daten entgegennehmen, sie gewichten und weiterleiten kann.

Aber ich muss ehrlich sein: Aktuell ist mein Netzwerk noch strunzdumm. Es kann nur lineare Mathematik, also gerade Linien ziehen. Es ist wie ein Auto ohne Motor – es hat alle Teile, sieht toll aus, aber es kann noch nirgendwohin fahren und keine komplexen Muster (wie handgeschriebene Zahlen) erkennen.