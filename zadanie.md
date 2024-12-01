Zadanie:
Tvojou úlohou je napísať funkčný Python kód pre riešenie regresnej úlohy. Máš prístup k trom súborom:

X_public.npy: Numpy súbor, ktorý obsahuje matice s príznakmi dát o rozmere NxF, kde N je počet vzoriek a F je počet príznakov pre každú vzorku.
y_public.npy: Numpy súbor obsahujúci vektor odpovedí (hodnôt), ktoré zodpovedajú dátam v X_public, o rozmere N prvkov.
X_eval.npy: Numpy súbor obsahujúci matice dát o rozmere Ne x F, ktoré slúžia na predikciu. Tvoj model má predikovať výstup (hodnoty) pre tieto dáta.
Tvoj cieľ:
Načítať tréningové dáta z X_public.npy a y_public.npy, následne natrénovať regresný model na týchto dátach. Môžeš použiť knižnice ako scikit-learn (prípadne iné modely po dohode).
Vytvoriť predikcie na dátach z X_eval.npy pomocou natrénovaného modelu a uložiť výsledný predikovaný vektor do súboru y_predikcia.npy.
Vyhodnotiť model pomocou metriky R² (r2_score) na tréningových dátach, aby si vedel, ako dobre tvoj model funguje.
Vygenerovať a odovzdať tieto výstupy:
Súbor y_predikcia.npy obsahujúci predikovaný vektor pre dáta X_eval.
Zdrojový kód tvojho riešenia v Python, v ktorom popíšeš, ako si model vytvoril a trénoval.
Krátku technickú správu, ktorá obsahuje:
Stručný teoretický opis regresného modelu, ktorý si použil.
Metodológiu riešenia, vrátane zdôvodnenia tvojich rozhodnutí (akú metódu si vybral a prečo).
Diskusiu o výsledkoch, vrátane porovnania rôznych prístupov, ktoré si mohol skúsiť.

Pre predikciu musiš pouzid mode Ridge. 
Cieľ je sa dostať nad 0.90 v R2 score.