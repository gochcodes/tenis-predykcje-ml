# ATP Tennis Match Predictor

Model uczenia maszynowego przewidujący wyniki meczów tenisa ziemnego w turniejach ATP. Projekt obejmuje pełen cykl Data Science: od czyszczenia danych i zaawansowanej inżynierii cech, przez trenowanie modelu, aż po interaktywny symulator szans.

**Wykorzystane technologie:** Python, Pandas, NumPy, Scikit-learn

## Główne funkcjonalności

Projekt składa się z dwóch kluczowych modułów:

1. **Inżynieria Cech i System Elo (datasety.ipynb)**
   * Przetwarzanie historycznych wyników meczów ATP (sortowanie chronologiczne).
   * Autorska implementacja algorytmu **Elo Rating**. Skrypt liczy siłę każdego gracza w ujęciu ogólnym oraz z podziałem na nawierzchnie (Hard, Clay, Grass, Carpet), uwzględniając wagę punktową turniejów (np. Grand Slam vs ATP250).
   * Śledzenie "streaków" – bieżącej passy wygranych/przegranych meczów z rzędu.

2. **Model Klasyfikacyjny i Symulator (predykcje.py)**
   * Wykorzystanie algorytmu **Random Forest Classifier** (100 drzew decyzyjnych) uczącego się nieliniowych zależności między różnicami w punktach Elo a rzeczywistym rezultatem.
   * Interaktywne środowisko w konsoli, które na bieżąco analizuje wprowadzone nazwiska zawodników i rodzaj kortu, zwracając procentowe prawdopodobieństwo zwycięstwa za pomocą funkcji predict_proba.

## Instrukcja uruchomienia

Aby przetestować symulator lokalnie, postępuj zgodnie z poniższymi krokami:

1. Sklonuj repozytorium i przejdź do folderu z projektem.
2. Zainstaluj wymagane biblioteki wpisując w terminalu:
   pip install pandas numpy scikit-learn
3. (Opcjonalnie) Uruchom komórki w pliku datasety.ipynb, aby od nowa wygenerować zaktualizowane pliki z rankingami Elo.
4. Uruchom skrypt głównego symulatora wpisując w terminalu:
   python predykcje.py
5. Postępuj zgodnie z instrukcjami w konsoli – wpisz nazwiska dwóch zawodników oraz pierwsza literę ich imienia zakończoną kropką (np. "Hurkacz H.", "Djokovic N.") oraz wybierz nawierzchnię, aby otrzymać wynik.

## Możliwości dalszego rozwoju (Roadmap)

Projekt stanowi solidną architekturę, którą można rozwijać o kolejne elementy:
* **Integracja Live API:** Zastąpienie statycznych plików .csv modułem pobierającym najnowsze wyniki bezpośrednio z serwisów sportowych, co pozwoli na ciągłą, w pełni automatyczną aktualizację rankingu Elo.
* **Rozszerzenie wektora cech:** Uwzględnienie w modelu statystyk H2H (bezpośrednich starć obu graczy) oraz wskaźnika zmęczenia fizycznego (łączny czas spędzony na korcie w poprzednich rundach turnieju).
* **Testowanie alternatywnych modeli:** Implementacja modeli opartych na gradiencie (np. XGBoost, LightGBM) i optymalizacja hiperparametrów za pomocą GridSearchCV.
* **Aplikacja Webowa:** Stworzenie graficznego interfejsu użytkownika z wykorzystaniem frameworka Streamlit do prostszej obsługi symulatora w przeglądarce.
