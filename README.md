# Simple Polish Comment Classification

## Opis aplikacji

Aplikacja wykorzystuje model BERT Multilingual Cased do klasyfikacji komentarzy w języku polskim. Model ten jest w stanie przewidywać następujące klasy:

- `toxicity`
- `severe_toxicity`
- `obscene`
- `threat`
- `insult`
- `identity_attack`
- `sexual_explicit`

Aplikacja analizuje wprowadzone komentarze i przypisuje im odpowiednie kategorie na podstawie ich treści.

## Uruchamianie aplikacji

Aby uruchomić aplikację przy pomocy Streamlit, wykonaj poniższe kroki:

1. Upewnij się, że masz zainstalowany Streamlit. Możesz go zainstalować za pomocą polecenia:
    ```bash
    pip install streamlit
    ```

2. Przejdź do katalogu, w którym znajduje się Twój plik aplikacji, np. `app.py`.

3. Uruchom aplikację za pomocą następującego polecenia:
    ```bash
    streamlit run app.py
    ```

4. Aplikacja zostanie uruchomiona w przeglądarce internetowej pod adresem `http://localhost:8501`.