# Fallstudie Model Engineering

Dieses Projekt wurde im Rahmen des Moduls "Fallstudie Model Engineering" des Studiengangs "Master Angewandte Data Science" erstellt.

## Projektbeschreibung

Dieses Projekt ist eine Fallstudie zum Modul "Fallstudie Model Engineering". Das Ziel ist die Entwicklung eines Modells zur Vorhersage des Erfolgs von Transaktionen für einen Zahlungsdienstleister (Payment Service Provider, PSP). Basierend auf diesen Vorhersagen werden verschiedene Geschäftsstrategien simuliert und bewertet, um die Kosten zu minimieren oder die Erfolgsrate zu maximieren.

## Projektstruktur

Das Projekt ist wie folgt strukturiert:

- **`/data`**: Enthält die Roh- und verarbeiteten Daten im CSV- und Excel-Format.
- **`/diagrams`**: Beinhaltet verschiedene Diagramme und Visualisierungen, die während der Analyse erstellt wurden.
- **`/models`**: Speichert die trainierten Machine-Learning-Modelle.
- **`/notebooks`**: Enthält Jupyter-Notebooks für die explorative Datenanalyse (`eda.ipynb`), den Hauptanalyse-Workflow (`main.ipynb`) und das Prototyping (`prototyping.ipynb`).
- **`/src`**: Beinhaltet den Python-Quellcode, der in folgende Module unterteilt ist:
    - `config.py`: Konfigurationsvariablen und -einstellungen.
    - `features.py`: Funktionen für das Feature Engineering.
    - `main.py`: Haupt-Skript zum Ausführen der Pipeline.
    - `metrics.py`: Funktionen zur Berechnung von Metriken und zur Evaluierung der Modelle.
    - `models.py`: Funktionen zum Trainieren der Modelle.
    - `predictions.py`: Funktionen zur Simulation und Bewertung von Geschäftsstrategien.
- `pyproject.toml`: Definiert die Projektabhängigkeiten und wird von `poetry` verwendet.

## Workflow

1.  **Daten laden und vorverarbeiten**: Die Daten werden aus `data/data.xlsx` geladen, Duplikate werden entfernt und die Daten werden in Trainings- und Testsets aufgeteilt.
2.  **Feature Engineering**: Aus den vorhandenen Daten werden neue Merkmale generiert. Dazu gehören kategoriale Merkmale, zyklische Merkmale aus Zeitstempeln und Merkmale, die sich auf wiederholte Transaktionsversuche beziehen.
3.  **Modelltraining**: Es werden verschiedene Modelle trainiert, darunter ein `DecisionTreeClassifier` und ein `HistGradientBoostingClassifier`. Der `HistGradientBoostingClassifier` wird zusätzlich mittels `RandomizedSearchCV` optimiert. Die trainierten Modelle werden im Verzeichnis `models` gespeichert.
4.  **Modellevaluierung**: Die Modelle werden anhand verschiedener Metriken wie Precision, Recall, F1-Score und ROC AUC bewertet. Zusätzlich wird die Merkmalswichtigkeit berechnet und visualisiert.
5.  **Simulation von Geschäftsstrategien**: Das Projekt simuliert verschiedene Geschäftsstrategien (kostenoptimiert, erfolgsoptimiert) und vergleicht diese mit dem Altsystem.

## Voraussetzungen

# Technologien

Software

- Python 3.10+
- poetry
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter

### Installation von Poetry

`pip install poetry`

### Projekt installieren und ausführen

Schritt 1 – Repository klonen

`git clone https://github.com/Erik01010/fallstudie_model_engineering`

Zum Pfad navigieren

`cd fallstudie_model_engineering`

Schritt 2 - Abhängigkeiten installieren

`poetry install`
`poetry env activate`
