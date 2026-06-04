### Tag 1: Der Bauplan – Was genau baue ich hier eigentlich?
Willkommen zu Tag 1! Bevor ich direkt in den Python-Code springe und wir uns in NumPy-Matrizen verlieren, wollte ich heute erstmal klären, was genau ich hier eigentlich baue. Wenn man so ein Projekt anfängt, muss man ja erstmal den Bauplan verstehen.

### Die Grundidee: Was ist ein Neuronales Netzwerk?
Normalerweise programmieren wir mit festen Regeln: "Wenn A passiert, mache B". Neuronale Netze sind da anders. Sie lernen durch Beispiele, ein bisschen wie unser eigenes Gehirn (naja, sehr vereinfacht gesagt).

Stellt es euch wie ein Fließband vor: Ich werfe vorne ein Bild rein (z. B. eine handgeschriebene Zahl), in der Mitte passieren viele mathematische Berechnungen, und hinten kommt eine Vorhersage raus, wie: "Das ist eine 7!".

### Das Herzstück: Das Neuron
Die kleinste Einheit auf diesem Fließband ist das Neuron. Vergesst die Biologie – im Code ist das einfach nur eine mathematische Funktion. Es nimmt Eingaben, verrechnet sie und gibt das Ergebnis weiter.

Das braucht drei Dinge:

Inputs: Die Daten, die reinkommen.

Gewichte (Weights): Wie wichtig ist diese Eingabe? Das Netz lernt mit der Zeit, diese "Wichtigkeits-Regler" richtig einzustellen.

Bias: Das ist eine Art Grundwert. Manchmal muss das Neuron ein Signal weitergeben, auch wenn der Input 0 ist. Der Bias schiebt das Ergebnis einfach ein Stück nach oben oder unten.

### Die Architektur: Mein Fließband
Wenn ich jetzt viele von diesen Neuronen verbinde, entsteht die Architektur. Für mein MNIST-Projekt (wo das Netz Zahlen von 0 bis 9 erkennen soll) baue ich ein Netz mit drei Schichten:

!(image.png)

Eingabeschicht (Input Layer): Hier kommen die Bilder rein. Meine MNIST-Bilder sind 28x28 Pixel groß. Das macht genau 784 Pixelwerte. Also hat meine erste Schicht exakt 784 Eingänge.

Verborgene Schichten (Hidden Layers): Hier passiert die eigentliche Arbeit. Diese Schichten suchen nach Mustern in den Pixeln (wie Kanten oder Kurven). Man kann selbst entscheiden, wie viele Schichten und Neuronen man hier nutzt.

Ausgabeschicht (Output Layer): Hier kommen die Ergebnisse an. Da es 10 mögliche Zahlen gibt (0 bis 9), brauche ich hier 10 Neuronen. Das Neuron, das am Ende den höchsten Wert hat, ist die Entscheidung des Netzwerks.

### Was macht das Netz am Ende wirklich schlau?
Wenn ich das nur so zusammenbaue, kann das Netz leider noch nicht viel. Es rechnet nur linear. Damit es wirklich "schlau" wird und komplexe Muster versteht, brauche ich noch drei Dinge, die ich in den nächsten Tagen Schritt für Schritt programmieren werde:

## Aktivierungsfunktionen: 
Sie bringen einen "Knick" in die Mathematik (Nicht-Linearität). Die bekanntesten sind ReLU (setzt negative Werte einfach auf 0, positive Werte bleiben unverändert) und Softmax (macht am Ende aus den Werten schöne Wahrscheinlichkeiten in Prozent, die die Summe aller Wahrscheinlichkeiten zu 100% ergibt).

## Verlustfunktion (Loss):
Am Anfang rät das Netz nur blind. Der Loss berechnet, wie falsch das Netz lag. Mein Ziel ist es, diesen Fehler durch das Training so nah wie möglich an 0 zu bringen. Je kleiner der Loss, desto besser.

### Backpropagation: 
Das ist die eigentliche Magie. Wenn das Netz falsch liegt, rechne ich rückwärts durch die Schichten, um zu schauen, welches Gewicht schuld war. Dann passe ich die Zahlen ein kleines bisschen an. Wenn ich das tausende Male mache, wird das Netz immer schlauer. Dieses Teil ist der schwierigste. Ich versuche es so einfach wie möglich zu erklären, damit es wirklich jeder versteht. Das ist das Herzstück, das die Magie ermöglicht. Es ist wie ein Feedback-Loop, der dem Netz sagt: "Hey, du hast falsch geraten, hier ist der Fehler, korrigiere dich!"

## Fazit für heute
So, das war die Theorie! Wenn man das erstmal verstanden hat, ist der nächste Schritt, das Ganze in Code zu gießen. 
Bei einer Projekt wie diesem ist es super wichtig, die Grundlagen zu verstehen, Bauplan zu kennen, bevor man loslegt. In den nächsten Tagen werde ich genau das tun: Schritt für Schritt die Theorie in Python-Code umsetzen, damit wir am Ende ein funktionierendes Neuronales Netzwerk haben, das MNIST-Zahlen erkennen kann.
