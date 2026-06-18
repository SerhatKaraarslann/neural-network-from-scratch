### Tag 5 : Backpropagation – Wir lernen aus Fehlern (Rückwärtsdurchlauf)
Willkommen zu Tag 5! In den Tag haben wir mit der Verlustfunktion unserem Netzwerk ein strenges Zeugnis ausgestellt. Wir wissen jetzt genau, wie schlecht unser Modell rät. Aber ein schlechtes Zeugnis zu bekommen reicht allein nicht,unser Netzwerk muss jetzt daraus lernen!

Heutiges Teil ist wie alles umkehren. Unser Fließband läuft jetzt komplett rückwärst. Der Fehlerrateberechnung, den wir gestern programmiert haben, wird nun von hinten nach vorne durch das gesamte Netzwerk gereicht, um herauszufinden, wer genau an dem Fehler schuld war. In der KI-Welt nennt man diesen Prozess Backpropagation.

Das klingt wie ein fieses Monster aus der Mathematik Hölle, weil es auf Differentialrechnung (Ableitungen) basiert. Aber eigentlich ist es nur wie das Spiel "Stille Post" – bloß rückwärts gespielt. Jede Station in meinem Code bekommt dafür jetzt eine eigene backward-Methode.

Ich habe heute meine drei Dateien (loss.py, layer.py und activations.py) aktualisiert. Schauen wir uns an, was ich da im Detail programmiert habe.

## Teil 1: Der Startschuss und der magische Mathe-Hack (Loss & Softmax)
Beim Spiel "Stille Post" muss jemand anfangen. Wir starten ganz am Ende unseres Fließbands. Dort stehen unsere Verlustfunktion (Categorical Cross-Entropy) und unsere letzte Aktivierungsfunktion (Softmax).

Eigentlich müssten wir jetzt eklige Ableitungen programmieren und riesige Matrizen miteinander verrechnen. Vor allem die Ableitung von Softmax allein ist mathematisch ein absoluter Albtraum.
Aber in meiner activations.py steht in der backward-Methode von Softmax einfach nur pass. Warum habe ich die leer gelassen? Hab ich was vergessen?

Nein, hier kommt mein magischer Mathe-Hack:
Findige Mathematiker haben herausgefunden, dass etwas Wundervolles passiert, wenn man die Rückwärtsrechnung von Softmax und Cross-Entropy direkt miteinander kombiniert. Die ganze furchtbare Mathematik mit Brüchen und Summen kürzt sich fast komplett weg! Übrig bleibt eine lächerlich einfache Formel:
Gradient = Vorhersage - Echte_Antwort
Deshalb habe ich diese Berechnung direkt in die loss.py gepackt.

Der Code in meiner Datei macht Folgendes:
- Ich nehme die Vorhersage (z.B. das Netz rät 90% bzw. 0.9 für die "7").
- Ich suche mir die Stelle der richtigen Antwort und ziehe dort einfach 1 ab (0.9 - 1.0 = -0.1).
- Alle anderen Vorhersagen bleiben unverändert.
- Dann teile ich das Ergebnis noch durch die Anzahl der Bilder in meiner Batch (self.dinputs = self.dinputs / samples), damit das Training stabil bleibt und die Fehlerwerte nicht explodieren.

Dieser berechnete Fehlerwert (der Gradient) ist unser Startschuss. Er wird jetzt rückwärts in die Dense Layer geworfen!

## Teil 2: Der Mechaniker (Die Dense Layer)
Unsere letzte Dense Layer empfängt nun diese Meldung von der Loss-Funktion: "Hey, wir lagen hier um den Wert X daneben!" Diesen reinkommenden Fehler nenne ich in meinem Code dvalues.

Jetzt muss meine Schicht in der layer.py den Mechaniker spielen und drei kritische Fragen beantworten:
1. Wie viel Schuld tragen meine Gewichte? (self.dweights)
Das ist die wichtigste Frage, denn die Gewichte wollen wir später anpassen! Um die Schuld zu finden, schauen wir uns die Eingabedaten an, die beim Vorwärtsdurchlauf in die Schicht reinkamen. Erinnert ihr euch? An Tag 2 habe ich im Code heimlich self.inputs = inputs geschrieben. Genau für diesen Moment!
Die Logik: Wenn ein Eingang damals 0 war, hat das dazugehörige Gewicht beim Multiplizieren nichts bewirkt und ist komplett unschuldig. War der Eingang aber hoch, ist das Gewicht stark schuld am Fehler.
Also rechne ich: Eingänge * Fehler. Im NumPy-Code löse ich das über ein Skalarprodukt und drehe die Eingangsmatrix um: np.dot(self.inputs.T, dvalues).

2. Wie viel Schuld tragen meine Biases? (self.dbiases)
Biases sind einfach. Ein Bias verstärkt nichts, er schiebt das Ergebnis nur stumpf nach oben oder unten. Daher ist er immer zu 100% für den Fehler verantwortlich, der bei ihm ankommt. Im Code summiere ich die Fehler also einfach nur Spalte für Spalte auf: np.sum(dvalues, axis=0, keepdims=True).

3. Welche Schuld reiche ich an die Schicht VOR mir weiter? (self.dinputs)
Beim Spiel "Stille Post" darf der Fehler nicht bei uns in der Ausgabeschicht stehenbleiben. Wir haben ja vielleicht noch eine weitere Dense Layer davor! Wir müssen der Station vor uns sagen, wie viel Mist sie gebaut hat. Dafür multiplizieren wir den erhaltenen Fehler mit unseren eigenen Gewichten und werfen das Ergebnis als dinputs weiter nach hinten.   

## Teil 3: Der Türsteher (ReLU)
Auf dem Weg weiter nach hinten trifft der Fehler auf unsere ReLU-Aktivierungsfunktion. Was hat ReLU beim Vorwärtsdurchlauf gemacht? Es hat alle negativen Zahlen auf 0 gesetzt.

In meiner activations.py habe ich die backward-Methode für ReLU geschrieben. Beim Rückwärtsdurchlauf funktioniert ReLU wie ein extrem strenger Türsteher. Es schaut sich den ankommenden Fehler an und gleicht ihn mit seinen alten Aufzeichnungen (self.input) ab:
"War dieses Neuron vorhin bei der Vorwärtsberechnung ausgeschaltet (0)?"

Wenn JA: Dann hat dieses Neuron gar nichts zum Endergebnis beigetragen! Also trifft es logischerweise auch keine Schuld am Fehler. Mein Türsteher-Code setzt den Fehler-Gradienten für dieses Neuron auf 0 (self.dinputs[self.input <= 0] = 0) und lässt ihn nicht weiter nach hinten durch.

Wenn NEIN: Dann war dieses Neuron aktiv und hat zum Endergebnis beigetragen. Es trägt also die volle Schuld am Fehler, der bei ihm ankommt. In diesem Fall lasse ich den Fehler-Gradienten unverändert weiter nach hinten durch.

## Fazit für heute
Heute haben wir die Backpropagation durch das gesamte Netzwerk programmiert. Wie ein Spiel "Stille Post" läuft der Fehler jetzt rückwärts durch die Schichten. Die Verlustfunktion gibt den Startschuss, die Dense Layer spielen den Mechaniker und die ReLU-Aktivierung ist der strenge Türsteher. Jetzt wissen wir nicht nur, wie schlecht unser Netzwerk rät, sondern auch genau, welche Neuronen und Gewichte an diesem Fehler schuld sind. 
Unser Netzwerk ist jetzt bereit, aus seinen Fehlern zu lernen und seine Gewichte anzupassen, um beim nächsten Mal besser zu raten!

Wir haben alle drei Dateien (loss.py, layer.py und activations.py) aktualisiert, um die Backpropagation zu implementieren. Jetzt können wir das gesamte Fließband vorwärts und rückwärts durchlaufen lassen, um unser Netzwerk zu trainieren. Hier kommt unser Fleißarbeit ins Spiel: Wir müssen jetzt die Gewichte anpassen, damit das Netzwerk beim nächsten Mal besser rät. Der Optimizer, unsere Methode zur Gewichtsaktualisierung!!!