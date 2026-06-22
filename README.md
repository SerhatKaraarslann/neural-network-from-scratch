# Neural Network from Scratch mit Pure Python und NumPy #

### Vom Konsumenten zum Konstrukteur : Ein Projekt über das Verstehen von Neuronalen Netzwerken/KI ###

Gestern habe ich auf YouTube ein Video gesehen (https://www.youtube.com/watch?v=cAkMcPfY_Ns), in dem ein Entwickler ein Neuronales Netzwerk komplett "from scratch" programmiert hat, also nur mit purem Python und der Mathematik-Bibliothek NumPy. Am Ende hat er es erfolgreich mit dem MNIST-Datensatz trainiert. Ich fand das extrem inspirierend, aber sein Code war leider hinter einer Paywall versteckt. Da dachte ich mir: *"Okay, warum baue ich es nicht einfach selbst? Wenn er das kann, kann ich das auch!"*

Ich bin Informatikstudent mit einem starken Fokus auf Künstliche Intelligenz. Wenn man in diesem Bereich tiefgreifendes Wissen erlangen möchte, ist es wichtig, die grundlegenden Konzepte und die Mathematik hinter Neuronalen Netzen zu verstehen. Theoretisch kenne ich die Funktionsweise und die Mathematik hinter Neuronalen Netzen. In der Uni nutzt man aber meist fertige Frameworks wie PyTorch oder TensorFlow. Das ist in der Praxis effizient, versteckt aber die eigentliche "Magie".
Aber das reicht nicht. Egal ob Large Language Models, Computer Vision oder andere Machine Learning Bereiche Neuronale Netze sind die Grundlage. Wenn ich wirklich verstehen will, wie sie funktionieren, muss ich sie selbst bauen.
Ich habe beschlossen, die Stützräder abzunehmen: Ich baue mein eigenes Netz komplett ohne fertige ML-Bibliotheken, um die Backpropagation, die Matrizen-Multiplikationen und die Architektur wirklich tiefgreifend zu meistern.

### Mein Ziel: Lernen durch Lehren (Die Feynman-Methode)
Wenn du etwas wirklich verstehen willst, versuche, es jemandem beizubringen. Genau deshalb existiert dieses Repository. Ich dokumentiere meine Reise Tag für Tag in meinem "Tagebuch", so geschrieben, als würde ich es einem guten Freund erklären. Kein KI-generierter Code für die Logik, sondern echte Handarbeit, Mathematik und ausführliche Kommentare.

### Was man hier findet?:
- Ein einfaches Neuronales Netzwerk, das von Grund auf neu erstellt wurde.
- Detaillierte Erklärungen zu jedem Schritt, von der Architektur über die Aktivierungsfunktionen bis hin zur Backpropagation.
- Ein Training auf dem MNIST-Datensatz, um die Leistungsfähigkeit des Netzes zu demonstrieren.

# Projektstruktur
Das Projekt ist in zwei Hauptbereiche unterteilt, ich habe es so gedacht und strukturiert, dass es leicht verständlich und nachvollziehbar ist:
- Den "Tagebuch"-Ordner, in dem ich meine täglichen Fortschritte dokumentiere
- Den "Code"-Ordner, in dem der eigentliche Code für das Neuronale Netzwerk liegt.


# Lokale Installation und Ausführung
Wenn du dir das Code nur anschauen möchtest, kannst du das direkt hier auf GitHub tun. Wenn du das Projekt lokal auf deinem Rechner ausführen willst, empfehle ich den blitzschnellen Python-Manager uv. Hier ist eine kurze Anleitung, wie du das Projekt lokal einrichten kannst:

1. Klone das Repository:
```bash
git clone https://github.com/SerhatKaraarslann/neural-network-from-scratch.git
cd neural-network-from-scratch
```
2. Virtuelle Umgebung erstellen:
Falls du uv noch nicht hast, installiere es (z.B. via curl -LsSf https://astral.sh/uv/install.sh | sh auf Mac/Linux oder powershell -c "irm https://astral.sh/uv/install.ps1 | iex" auf Windows).
```bash
uv venv 
```

3. Umgebung aktivieren:
```bash
Windows: .\.venv\Scripts\activate
Mac/Linux: source .venv/bin/activate 
```
4. Abhängigkeiten installieren:
```bash
uv add numpy
uv add tensorflow # Nur für den MNIST-Datensatz, kein TensorFlow-Code im Netzwerk selbst!
```
5. Das Neuronale Netzwerk ausführen:
```bash
 uv run ...
```

### Schlusswort
Ich hoffe, dass dieses Projekt nicht nur mir, sondern auch anderen dabei hilft, die Welt der Neuronalen Netzwerke besser zu verstehen. Meine Empfehlung : Führe den Code nicht nur aus! Geh in der Ordner /tagebuch/ und lies dir die täglichen Einträge durch, um wirklich zu verstehen, was hinter den Kulissen passiert. Viel Spaß beim Lernen und Entdecken!



