### Tag 6: Der Optimizer – Der Trainer, der unser Netzwerk fit macht

Willkommen zu Tag 6! An Tag 5 haben wir mit dem Backpropagation-Algorithmus gearbeitet. Wir haben das gesamte Netzwerk rückwärts durchlaufen und exakt berechnet, wie viel Schuld jedes einzelne Gewicht und jeder einzelne Bias am finalen Fehler (Loss) hatte.

Aber das reicht nicht! Wir haben die Gewichte noch gar nicht verändert! Wir wissen jetzt zwar, wer an dem Fehler schuld ist, aber wir müssen auch etwas dagegen tun. Wir müssen die Gewichte und Biases physisch anpassen, damit unser Netzwerk beim nächsten Mal besser rät. Genau hier kommt der Optimizer ins Spiel!

Der Optimizer ist wie ein Trainer, der unserem Netzwerk sagt: "Hey, du hast hier Mist gebaut, aber das ist okay! Lass es mich für dich reparieren!" Er nimmt die Schuldzuweisungen (die Gradienten) aus der Backpropagation und berechnet, wie die Gewichte und Biases angepasst werden müssen, um den Fehler zu reduzieren.
In meinem Code habe ich eine neue Datei optimizers.py erstellt, in der ich diese Logik für die Gewichtsaktualisierung programmiert habe.

## Teil 1: Der einfachste Trainer – Stochastic Gradient Descent (SGD)

Es gibt viele verschiedene Optimizer in der KI-Welt, wie Adam, RMSprop oder Adagrad. Aber heute starten wir mit dem einfachsten und klassischsten aller Optimizer: Stochastic Gradient Descent (SGD).

Um zu verstehen, wie SGD funktioniert, müssen wir uns Folgendes vorstellen: Wir stehen mit verbundenen Augen auf einem sehr hügeligen Berg. Unser Ziel ist es, ganz nach unten ins Tal zu kommen, denn ganz unten im Tal ist der Fehler (unser Loss) am geringsten, idealerweise gleich 0.
Aber wir können nicht sehen, wo wir hinlaufen, also müssen wir uns auf unser Gefühl verlassen. Wir tasten mit unseren Füßen den Boden ab und spüren, in welche Richtung es steil bergauf geht. Das ist genau das, was der Gradient (den wir in der Backpropagation berechnet haben) uns sagt: "Hey, in diese Richtung steigt der Fehler am stärksten an!"

Wenn wir wissen, wo es nach oben geht, dann wissen wir logischerweise auch, wo es nach unten geht. Der Gradient zeigt uns die Steigung, und wir müssen einfach in die entgegengesetzte Richtung gehen, um den Fehler zu reduzieren.

# Die Mathematik dahinter: Warum nutzen wir Minus?

Der Code für SGD in meiner Datei optimizers.py ist ziemlich einfach. Er nimmt die Gewichte und Biases, die Schuldzuweisungen (Gradienten) und die Lernrate als Eingabe und berechnet die neuen Gewichte und Biases.

Die Formel für die Gewichtsaktualisierung lautet:

layer.weights += -self.learning_rate * layer.dweights
layer.biases += -self.learning_rate * layer.dbiases


Warum rechnen wir hier mit einem Minus? Ein Gradient (dweights) zeigt mathematisch immer steil bergauf zum größten Fehler. Wenn wir den Fehler einfach addieren würden, würde unser Netzwerk mit jedem Schritt weiter in die falsche Richtung laufen und der Fehler würde immer größer werden. Das Minus kehrt die Richtung einfach um, sodass wir tatsächlich bergab zum Tal laufen, wo der Fehler am geringsten ist.

# Die Lernrate – Der Schlüssel zum Erfolg

Wie weit sollen wir in diese Richtung gehen? Das beantwortet die Lernrate (learning_rate). Sie ist wie ein Schrittmaß und einer der wichtigsten Parameter im Deep Learning! Die Lernrate bestimmt, wie groß unsere Schritte auf dem Berg sein sollen.

Ist die Lernrate zu groß (Riesenschritte): Dann könnten wir über das Ziel (das Tal) hinausschießen und auf der anderen Seite des Berges landen, wo der Fehler vielleicht sogar noch größer ist. Das Netzwerk lernt nichts, die Werte explodieren.

Ist die Lernrate zu klein (Ameisenschritte): Dann laufen wir zwar extrem sicher nach unten, aber es könnte ewig dauern, bis wir das Tal erreichen, und unser Netzwerk könnte in einer kleinen Mulde (einem lokalen Minimum) steckenbleiben, anstatt ganz unten anzukommen.

Oft startet man mit einer Lernrate von 1.0, 0.1 oder 0.01 und passt sie dann je nach Bedarf an.

## Fazit für heute

Heute haben wir den Optimizer programmiert, unseren Trainer, der unser Netzwerk fit macht. Mit Stochastic Gradient Descent (SGD) haben wir eine einfache, aber mächtige Methode implementiert, um die Gewichte und Biases unseres Netzwerks physisch anzupassen, basierend auf den Schuldzuweisungen aus der Backpropagation.

Wir haben verstanden, dass die Lernrate ein entscheidender Faktor ist, um sicherzustellen, dass unser Netzwerk effektiv lernt und nicht in die falsche Richtung läuft oder zu langsam vorankommt.

Jetzt haben wir wirklich alle Einzelteile zusammengebaut! Wir sind bereit für Tag 7: Morgen werden wir alle unsere Bausteine (Layer, Aktivierungsfunktionen, Loss und Optimizer) zu einer großen Netzwerk-Klasse zusammenfügen. Dann beginnt endlich das große Training!