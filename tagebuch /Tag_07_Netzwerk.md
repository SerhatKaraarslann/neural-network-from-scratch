### Tag 7: Das Netzwerk – Alle Bausteine vereint und bereit für das Training
Willkommen zu Tag 7! Heute is ein ganz besonderer Tag, denn heute werden wir alle unsere Bausteine – die Dense Layer, die Aktivierungsfunktionen, die Verlustfunktion und den Optimizer – zu einer großen Netzwerk-Klasse zusammenfügen. Das ist wie das Zusammenbauen eines riesigen Lego-Sets, bei dem jedes Teil perfekt ineinandergreift.

Nachdem wir jetzt alle Einzelteile programmiert haben, können wir sie endlich zu einem vollständigen Neuronalen Netzwerk zusammenfügen. In meiner Datei network.py habe ich eine neue Klasse namens Network erstellt, die all diese Komponenten integriert.

Bevor wir uns den neuen Code anschauen, ist der Moment, um einmal tief durchzuatmen und unsere Reise zu reflektieren. Was haben wir bisher gelernt? 

## Rückblick auf die Reise
1. Tag 1 - Die Architektur: Wir haben verstanden, wie ein neuronales Netzwerk aufgebaut ist. Wir haben die Grundstruktur unseres Netzwerks definiert, angefangen mit der Input-Schicht, über die versteckten Dense Layer bis hin zur Output-Schicht. 

2. Tag 2 - Die Dense Layer: Wir haben mit layer.py die Basis unseres Netzwerk gebaut. Eine Dense Layer ist eine Schicht aus künstliche Neuronen, die alle Eingänge mit Gewichten multiplizieren, einen Bias hinzufügen und das Ergebnis weitergeben. 

3. Tag 3 - Aktivierungsfunktionen (Nicht Linearität): Nur mit Dense Layers könnte unser Netzwerk nur dumme, gerade Linien ziehen. Mit der activations.py haben wir ReLU und Softmax programmiert, um unserem Netzwerk die Fähigkeit zu geben, komplexe Muster zu erkennen.
- ReLU (Rectified Linear Unit) setzt alle negativen Werte auf 0 und lässt positive Werte unverändert. Das bringt die notwendige Nicht-Linearität in unser Netzwerk.
- Softmax am Ende unserer Output-Schicht verwandelt die rohen Werte in Wahrscheinlichkeiten, die die Summe aller Ausgaben zu 100% ergeben.

4. Tag 4 - Verlustfunktion (Der strenge Lehrer): In der loss.py haben wir mit der Categorical Cross-Entropy Loss-Funktion einen strengen Lehrer programmiert, der unserem Netzwerk sagt, wie schlecht es rät. Je höher der Loss, desto schlechter die Vorhersage. Unser Ziel ist es, diesen Loss so weit wie möglich zu minimieren, damit unser Netzwerk immer besser wird.

5. Tag 5 - Backpropagation (aus Fehlern lernen): Der absolute Endgegner der KI-Welt! Wir haben das gesamte Netzwerk rückwärts durchlaufen und genau berechnet, wie viel Schuld jedes Gewicht und jeder Bias am Fehler trägt. Das ist die Grundlage dafür, dass unser Netzwerk überhaupt lernen kann.

6. Tag 6- Optimizer (Der Trainer): Wir wissen jetzt, wer an dem Fehler schuld ist, aber wir müssen auch etwas dagegen tun. Der Optimizer nimmt die Schuldzuweisungen aus der Backpropagation und berechnet, wie die Gewichte und Biases angepasst werden müssen, um den Fehler zu reduzieren. Mit Stochastic Gradient Descent (SGD) haben wir eine einfache Methode implementiert, um die Gewichte physisch anzupassen.

## Der große Moment: Das Netzwerk vereint
Jetzt, da wir alle Bausteine haben, können wir sie endlich in unserer Network-Klasse zusammenfügen. In network.py habe ich die Klasse NeuralNetwork geschrieben.
Wenn ich diese Klasse starte, baut sie im Hintergrunf automatisch mein gesamtes Netzwerk auf.
Das Netzwerk besteht aus:
- Eingang : 784 Pixel (28x28 MNIST-Bilder)
- Hidden Layer 1 : Eine Layer_Dense mit 128 Neuronen, gefolgt von ReLU
- Hidden Layer 2 : Eine weitere Layer_Dense mit 64 Neuronen, gefolgt von ReLU
- Ausgang : Eine Layer_Dense mit 10 Neuronen, gefolgt von Softmax
- Loss : Categorical Cross-Entropy
- Optimizer : Stochastic Gradient Descent (SGD)

# Der Trainingsloop – Das Herzstück des Lernens
Das Herzstück unseres Netzwerks ist der Trainingsloop. Hier passiert die Magie! Wenn wir diese Methode train_step() aufrufen, läuft das gesamte Fließband vorwärts und rückwärts durch unser Netzwerk:
1. Vorwärtsdurchlauf: Das Bild wird durch die Dense Layer und Aktivierungsfunktionen geschleust, bis am Ende eine Vorhersage herauskommt.
2. Verlustberechnung: Die Loss-Funktion vergleicht die Vorhersage mit der echten Antwort und berechnet den Fehler (Loss).
3. Rückwärtsdurchlauf: Die Backpropagation läuft rückwärts durch das Netzwerk, um zu berechnen, welche Gewichte und Biases an diesem Fehler schuld sind.
4. Gewichtsaktualisierung: Der Optimizer passt die Gewichte und Biases basierend auf den Schuldzuweisungen an, damit das Netzwerk beim nächsten Mal besser rät.


## Fazit für heute
Heute haben wir alle unsere Bausteine zu einer großen Netzwerk-Klasse zusammengefügt. Wir haben einen vollständigen Trainingsloop programmiert, der es unserem Netzwerk ermöglicht, aus seinen Fehlern zu lernen und seine Gewichte anzupassen, um immer bessere Vorhersagen zu machen. Jetzt sind wir bereit, unser Netzwerk auf dem MNIST-Datensatz zu trainieren und zu sehen, wie gut es wirklich ist! Morgen werden wir genau das tun: Wir werden unser Netzwerk mit echten Bildern füttern und beobachten, wie es lernt, handgeschriebene Zahlen zu erkennen. Das ist der Moment, auf den wir alle hingearbeitet haben!
