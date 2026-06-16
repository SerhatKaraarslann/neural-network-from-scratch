### Tag 3: Aktivierungsfunktionen - Der Motor meines Netzwerks
Willkommen zu Tag 3! In dem Tag 2 haben wir unsere Dense Layer gebaut. Das Fließband steht, die Neuronen sind verbunden, aber das Netz ist immer noch strunzdumm. Es kann nur lineare Muster erkennen. Heute fügen wir den Motor hinzu, damit unser Netzwerk wirklich "schlau" wird: Die Aktivierungsfunktionen.

### Warum Aktivierungsfunktionen so wichtig sind
Unser Problem ist, dass die Ausgabe unserer Schicht immer eine lineare Kombination der Eingänge ist. Wenn ich einfach hunderte von meinem Dense Layers aus Tag 2 hinterinanderhänge, passiert immer noch das Gliech. Eine lineare Gleichung plus eine lineare Gleichung ergibt immer noch eine lineare Gleichung. Das Netz könnte also nur gerade Linien ziehen, egal wie viele Schichten ich baue. Das ist wie ein Auto ohne Motor – es hat alle Teile, sieht toll aus, aber es kann nirgendwohin fahren.

Egal wie tief mein Netzwerk ist, könnte es nur lineare Muster erkennen. Wenn die Daten aber komplexe Muster haben (wie handgeschriebene Zahlen), dann ist das ein Problem. Um komplexe Muster zu erkennen, brauche ich Nicht-Linearität. Und genau das bringen Aktivierungsfunktionen ins Spiel. Sie fügen einen "Knick" in die Mathematik ein, der es dem Netzwerk ermöglicht, komplexe Muster zu lernen und zu erkennen.

### ReLU : Knick in der Mitte
Für die Hidden Layers nutze ich die ReLU. Es ist die industrie Standart für Hidden Layers. ReLU steht für "Rectified Linear Unit". Es ist eine super einfache Funktion: 
ReLU(x) = max(0, x)
- Ist der Input negativ? Setze ihn auf 0.
- Ist der Input positiv? Lass ihn unverändert.

Es ist so einfach aber sehr effektiv. ReLU hilft dem Netzwerk, komplexe Muster zu lernen, ohne dass die Berechnungen zu kompliziert werden. Ein Neuron, das ReLU nutzt und ein negatives Ergebnis hat, gibt einfach 0 aus. Das ist wie ein Schalter, der das Neuron ausschaltet. Wenn das Neuron ein positives Ergebnis hat, gibt es dieses Ergebnis weiter. Das ermöglicht es dem Netzwerk, wichtige Muster zu erkennen und unwichtige Informationen zu ignorieren. Blitzschnell und super effektiv!

### Softmax : Wahrscheinlichkeiten am Ende
Die ReLU ist super für die Hidden Layers aber für Output Layer brauche ich etwas anderes. Da brauche ich nicht 0 oder positive Werte, sondern ich will am Ende eine Wahrscheinlichkeitsverteilung über meine 10 möglichen Zahlen (0-9).Mein Netzwerk soll Ziffern von 0 bis 9 erkennen. Am Ende spucken meine 10 Neuronen völlig wilde Zahlen aus, z.B. [4.5, -1.2, 12.8, 0.1, ...]. Was soll ich damit anfangen? Ich kann ja schlecht sagen: "Das Bild ist mit einer Wahrscheinlichkeit von 12,8 eine Zwei.

Hier kommt die Softmax ins Spiel. Softmax nimmt diese wilden Zahlen und verwandelt sie in Wahrscheinlichkeiten, die zwischen 0.0 und 1.0 liegen und deren Summer exact 100% ergibt. 

Softmax(x_i) = exp(x_i) / sum(exp(x_j)) für alle j
- exp(x_i) ist die Exponentialfunktion. Es nimmt die exponentielle von jedem Wert. Das sorgt dafür, dass alle Werte positiv werden und größere Werte noch größer erscheinen.
- Dann teilt es jeden exponentiellen Wert durch die Summe aller exponentiellen Werte. Das sorgt dafür, dass die Ausgabe eine gültige Wahrscheinlichkeitsverteilung ist. Bingo! Wir haben saubere Wahrscheinlichkeiten, die uns sagen, wie wahrscheinlich es ist, dass das Bild eine 0, 1, 2, ... oder 9 ist. Das Neuron mit der höchsten Wahrscheinlichkeit gewinnt und sagt: "Das ist eine 7!"

### Geht das auch ohne ReLU und Softmax?
Technisch gesehen könnte ich auch ohne diese Funktionen arbeiten, aber das wäre wie ein Auto ohne Motor zu fahren. Es würde einfach nicht funktionieren. Ohne ReLU könnte mein Netzwerk nur lineare Muster erkennen, und ohne Softmax könnte ich keine sinnvollen Vorhersagen am Ende machen. Es wäre wie ein Auto zu bauen, das nur geradeaus fahren kann und am Ende nicht sagen kann, ob es ankommt oder nicht.

### Geheimtrick in der Code - Stabilität bei Softmax
Wenn ihr euch meinen Code für Softmax genauer anschaut, fällt euch vllt. diese seltsame Zeile auf: exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)). Das ist ein kleiner Trick, um die Berechnung stabiler zu machen. Die exponentielle Funktion wächst rasend schnell. Wenn das Netwerk am Anfang wild rät und eine große Zahl ausspuckt wie 100, würde Python versuchen exp(100) auszurechnen. Die Zahl ist so groß, dass das Programm abstürzt.

Ich ziehe also von allen Eingängen den größten Wert ab, bevor ich die Exponentialfunktion anwende. So wird die größte Zahl exponentiell zu 1 (weil exp(0) = 1) und alle anderen Zahlen werden kleiner. Mein Program bleibt stabil, stürzt nicht ab, und ich bekomme trotzdem die richtigen Wahrscheinlichkeiten am Ende.

### Fazit für heute
Ta da!!! Jetzt hat das Netz einen Motor! Wir können Daten durch die Danse Layer schicken und sie mit ReLU in den Hidden Layers verarbeiten, und am Ende mit Softmax eine saubere Wahrscheinlichkeitsverteilung über die möglichen Zahlen bekommen. Die Vorwärstausbreitung ist jetzt komplett!

Wenn ich jetz ein Bild durch das Netz schicke,rechnet es, denkt nicht mehr linear und gibt mir am Ende schöne Prozentzahlen aus.

Noch habe ich ein Problem und zwar die Startwerte. Die sind zufällig und das Netz rät einfach Müll. Aber das ist okay, denn morgen fügen wir die Verlustfunktion und die Backpropagation hinzu, damit das Netz aus seinen Fehlern lernen und immer bessere Vorhersagen machen kann. 