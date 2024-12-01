Krok 1: Inicializácia projektu a príprava prostredia
Prompt:

Vytvor súbor requirements.txt a pridaj knižnice: numpy, scikit-learn.
Vytvor súbor main.py a priprav základnú štruktúru kódu:
Importuj potrebné knižnice.
Pridaj komentár popisujúci zadanie projektu.
Vytvor Python skript na inštaláciu závislostí:
bash
Copy code
python -m pip install -r requirements.txt
Krok 2: Načítanie dát
Prompt: Doplň do súboru main.py:

Kód na načítanie dát zo súborov X_public.npy a y_public.npy.
Skontroluj rozmery načítaných matíc a vektora.
Uisti sa, že sú pripravené na použitie v modeli.
Príklad:

python
Copy code
import numpy as np

# Načítanie dát
X_public = np.load("X_public.npy")
y_public = np.load("y_public.npy")

# Kontrola rozmerov
print("Shape of X_public:", X_public.shape)
print("Shape of y_public:", y_public.shape)
Krok 3: Trénovanie modelu Ridge
Prompt: Doplň do main.py:

Import modelu Ridge zo scikit-learn.
Nastavenie modelu s vhodným parametrom regularizácie (alpha).
Trénovanie modelu na dátach X_public a y_public.
Príklad:

python
Copy code
from sklearn.linear_model import Ridge

# Inicializácia a trénovanie modelu
model = Ridge(alpha=1.0)
model.fit(X_public, y_public)

print("Model trained successfully.")
Krok 4: Vyhodnotenie modelu pomocou R²
Prompt: Doplň do main.py:

Import metriky r2_score.
Vyhodnotenie modelu na tréningových dátach a výpis skóre.
Príklad:

python
Copy code
from sklearn.metrics import r2_score

# Predikcia na tréningových dátach
y_train_pred = model.predict(X_public)

# Výpočet R² skóre
r2 = r2_score(y_public, y_train_pred)
print("R² score on training data:", r2)
Krok 5: Predikcia na evaluačných dátach
Prompt: Doplň do main.py:

Načítaj evaluačné dáta X_eval.npy.
Predikuj výstupy pre tieto dáta.
Ulož výsledok do y_predikcia.npy.
Príklad:

python
Copy code
# Načítanie evaluačných dát
X_eval = np.load("X_eval.npy")

# Predikcia
y_predikcia = model.predict(X_eval)

# Uloženie predikcií
np.save("y_predikcia.npy", y_predikcia)
print("Predictions saved to y_predikcia.npy.")