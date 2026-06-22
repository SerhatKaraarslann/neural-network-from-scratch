### Tag 8 : Das Training – Unser Netzwerk lernt, handgeschriebene Zahlen zu erkennen
Willkommen zu Tag 8! Das ist der Tag, auf den wir die ganze Zeit hingearbeitet haben. Wir haben die Architektur entworfen, die Dense Layers gebaut, den Fehler-Bewerter programmiert, unseren Optimizer eingestellt und unsere gesamte Fabrik (NeuralNetwork) zusammengesetzt.

Heute werden wir unser Netzwerk mit echten Daten füttern und beobachten, wie es lernt, handgeschriebene Zahlen zu erkennen. Wir werden den Trainingsloop implementieren, der es unserem Netzwerk ermöglicht, aus seinen Fehlern zu lernen und seine Gewichte anzupassen, um immer bessere Vorhersagen zu machen.

## Ein kleines Geständnis vorwer : Warum TensorFlow?
Ganz am Anfang dieses Porjekts habe ich eine strenge Regel aufgestellt: Wir bauen alles from scratch, nur mit Python und numpy! Wenn wir jetzt aber meinen Code in der main.py anschauen, werden wir sehen, dass ich dort TensorFlow importiere (from tensorflow.keras.datasets import mnist). Ich habe da Tensorflow nur als bequemere Lieferdienst für den MNIST-Datensatz verwendet, damit wir nicht selbst die Bilder herunterladen und vorverarbeiten müssen.

## Der Treibstoff für unser Netzwerk: Der MNIST-Datensatz
Der MNIST-Datensatz ist eine Sammlung von 70.000 handgeschriebenen Zahlenbildern, die in 10 Klassen (0-9) unterteilt sind. Es ist der "Hello World" der Computer Vision und ein perfekter Test für unser Netzwerk. Jedes Bild ist 28x28 Pixel groß, was bedeutet, dass wir 784 Eingänge für unser Netzwerk haben (28 * 28 = 784).
In main.py habe ich den MNIST-Datensatz geladen und in Trainings- und Testdaten aufgeteilt. Die Trainingsdaten werden verwendet, um unser Netzwerk zu trainieren, während die Testdaten dazu dienen, die Leistung unseres Netzwerks zu bewerten, nachdem es trainiert wurde.

Um die Bilder für unser Netzwerk vorzubereiten, habe ich eine Hilfsfunktion(preprocess_data)verwendet, um sie in einen 1D-Vektor umgewandelt (von 28x28 auf 784) und die Pixelwerte normalisiert (durch 255 geteilt), damit sie zwischen 0 und 1 liegen. Das hilft unserem Netzwerk, schneller zu lernen.

Um alles Object-Orientiert zu halten, habe ich eine Klasse Trainer erstellt, die den gesamten Trainingsprozess kapselt. In dieser Klasse habe ich eine Methode train() definiert, die den Trainingsloop implementiert.

## Der Trainingsloop – Das Herzstück des Lernens
Der Trainingsloop ist der magische Moment, in dem unser Netzwerk tatsächlich lernt. Hier passiert die ganze Magie! Wenn wir die Methode train() aufrufen, läuft das gesamte Fließband vorwärts und rückwärts durch unser Netzwerk:
1. Vorwärtsdurchlauf: Das Bild wird durch die Dense Layer und Aktivierungsfunktionen geschleust, bis am Ende eine Vorhersage herauskommt.
2. Verlustberechnung: Die Loss-Funktion vergleicht die Vorhersage mit der echten Antwort und berechnet den Fehler (Loss).
3. Rückwärtsdurchlauf: Die Backpropagation läuft rückwärts durch das Netzwerk, um zu berechnen, welche Gewichte und Biases an diesem Fehler schuld sind.
4. Gewichtsaktualisierung: Der Optimizer passt die Gewichte und Biases basierend auf den Schuldzuweisungen an, damit das Netzwerk beim nächsten Mal besser rät.

In main.py habe ich die train() Methode so implementiert, dass sie über eine bestimmte Anzahl von Epochen läuft (die Anzahl der Durchläufe durch den gesamten Trainingsdatensatz). In jeder Epoche wird der Trainingsloop für jedes Bild im Trainingsdatensatz ausgeführt, und am Ende jeder Epoche berechnen wir die durchschnittliche Genauigkeit und den Verlust, um zu sehen, wie gut unser Netzwerk lernt.

## Der Schockmoment: Mein Buchstabendreher Bug
Als ich das Training zum ersten Mal startete, passierte etwas Verrücktes. Die Konsole zeigte mir während des Trainings fantastische Werte:
Step 400/468 - Loss: 0.4081 - Accuracy: 85.16%

Das Netzwerk hat also zu 85% richtig geraten! Aber dann kam die Abschlussprüfung auf den unbekannten Testdaten (trainer.test) und ich traute meinen Augen nicht:
Finale Genauigkeit auf den Testdaten: 32.90%
Finaler Verlust auf den Testdaten: 0.9069

Wie konnte das Netz im Training so schlau und im Test plötzlich so dumm sein? 32% ist fast reines, blindes Raten!
Ich suchte verzweifelt nach Fehlern in der Mathematik, bis ich den wahren Übeltäter fand. Es war kein KI-Problem, sondern ein simpler Python-Tippfehler!

In meiner Methode evaluate gab ich die Werte so zurück: return accuracy, loss.
Aber in meiner main.py fing ich sie so auf: loss, accuracy = self.network.evaluate(...).

Python hat die beiden Zahlen einfach vertauscht! Mein Netzwerk hatte in Wahrheit eine fantastische Genauigkeit von 90.69% und einen Loss von 0.3290. Weil die Reihenfolge vertauscht war, dachte mein Print-Befehl: "Ah, Accuracy ist 0.3290 (32%)!". Das Netz war die ganze Zeit extrem schlau, mein Code hat das Ergebnis nur falsch herum auf den Bildschirm gedruckt! Das war ein echter Schockmoment, aber auch eine wichtige Lektion: Manchmal liegt der Fehler nicht in der komplexen KI-Mathematik, sondern in einem einfachen Tippfehler. Es ist immer wichtig, den Code sorgfältig zu überprüfen und sicherzustellen, dass alle Variablen korrekt verwendet werden.

## Mein experimenteller Trainingsverlauf
Nachdem der Bug behoben war, habe ich exprementiert. Zuerst dachte ich mir: "Je mehr Neuronen/Layer, desto intelligenter das Netzwerk!". Ich hatte mein Netzwerk mit zwei Hidden Layern (128 und 64 Neuronen) gebaut. Das Ergebnis war okay, nicht überragend.

Im Training: Step 400/468 - Loss: 0.8576 - Accuracy: 73.44%
Abschlussprüfung (Test): Finale Genauigkeit: 74.71% | Verlust: 0.7866

Knapp 75% ist schon ziemlich gut, aber ich wollte mehr! Ich habe dann etwas recherhiert und eine goldene Regel des Machine Learnings verstanden : Mehr Layer bedeutet nicht automatisch, dass das Netzwerk besser lernt! Manchmal machen zu viele Lernen sogar schwerer, weils sich der Fehler bei der Backpropagation über viele Stationen hinweg verliert (Vanishing Gradient Problem). Besonders bei so simplen Datensätzen wie MNIST reichen oft schon ein oder zwei Hidden Layer, um eine sehr gute Leistung zu erzielen.
Also habe ich meinen Code angepasst und die Arhitektur meines Netzwerks vereinfacht. Ich habe den zweiten Hidden Layer entfernt und nur einen Hidden Layer mit 128 Neuronen verwendet. . Ich startete das Training neu... und wow:

Im Training: Step 400/468 - Loss: 0.4119 - Accuracy: 86.72%
Abschlussprüfung (Test): Finale Genauigkeit: 90.70% | Verlust: 0.3287

Über 90% Genauigkeit!
Der Loss hat sich massiv verringert und die Trefferquote ist durch die Decke geschossen. Das einfachere Netzwerk hat die Muster in den Zahlen viel schneller und präziser verstanden. Für den MNIST-Datensatz reicht ein Hidden Layer absolut aus.

### Fazit für heute
Heute haben wir unser Netzwerk mit echten Daten gefüttert und beobachtet, wie es lernt, handgeschriebene Zahlen zu erkennen. Wir haben den Trainingsloop implementiert, der es unserem Netzwerk ermöglicht, aus seinen Fehlern zu lernen und seine Gewichte anzupassen, um immer bessere Vorhersagen zu machen. Wir haben auch einen wichtigen Bug gefunden und behoben, der uns glauben ließ, dass unser Netzwerk viel schlechter war, als es tatsächlich war. Schließlich haben wir experimentiert und gelernt, dass mehr Layer nicht immer besser ist, manchmal ist weniger mehr! 

Es ist vollbracht! Aus einer leeren Python-Datei haben wir ein komplettes Neuronales Netzwerk gebaut, das in der Lage ist, handgeschriebene Zahlen mit über 90% Genauigkeit zu erkennen. 
Im Tag 9 werde ich einen anderen Datensatz verwenden, und zwar den Fashion-MNIST-Datensatz, der Bilder von Kleidungsstücken enthält. Mal sehen, ob unser Netzwerk auch dort so gut performt!