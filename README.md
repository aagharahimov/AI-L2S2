# Disaster Tweet Analysis Dashboard

Analyzes tweets for disaster-related needs, urgency, sentiment, and location using AI (DistilBERT) and displays results on an interactive dashboard (Flask backend, React frontend).

## Key Features

* **AI Analysis:** DistilBERT fine-tuned for need classification (flood, fire, etc.), VADER sentiment analysis, keyword-based urgency scoring.
* **Location Mapping:** Extracts city names (using GeoNames) and plots approximate coordinates on a Leaflet map.
* **Interactive UI:** Sortable table of results, map visualization with category icons, click-to-pan map functionality.
* **Data Preprocessing:** Script included (`add_coordinates.py`) to enrich data with coordinates.

## Technologies

* **Backend:** Python, Flask, Pandas, NLTK, Transformers (Hugging Face), PyTorch, Scikit-learn, PyCryptodome
* **Frontend:** React.js, Axios, Leaflet, React-Leaflet, CSS
* **Data:** CSV, GeoNames (`cities15000.txt`)

## Prerequisites

* Python 3.x
* Node.js and npm (or yarn)

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/aagharahimov/AI-L2S2
    cd disaster_response_sim
    ```

2.  **Backend Setup:**
    * **Create & Activate Virtual Environment:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        # On Windows use: venv\Scripts\activate
        ```
    * **Install Python Dependencies:**
        ```bash
        pip install Flask Flask-CORS pandas nltk transformers torch scikit-learn pycryptodome
        ```
    * **Download NLTK Data (if first time):**
        ```python
        # Run this in Python interpreter within the activated venv
        import nltk
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        exit()
        ```
    * **Download Fine-Tuned Model:**
        * Download the `fine_tuned_bert` model files from:
            **https://seafile.unistra.fr/d/c684b802516d4c6b8e87/**
        * Create `disaster_response_sim/fine_tuned_bert/` folder.
        * Place downloaded model files inside it.

    * **Get GeoNames Data:**
        * Download `cities15000.zip` from [http://download.geonames.org/export/dump/](http://download.geonames.org/export/dump/).
        * Unzip and place `cities15000.txt` in the main `disaster_response_sim` directory.

    * **Prepare Data:**
        * Ensure `train.csv` is in the main directory.
        * Run the preprocessing script (this creates `train_with_coords.csv`):
            ```bash
            python add_coordinates.py
            ```

3.  **Frontend Setup:**
    * ```bash
        cd frontend
        npm install
        # or: yarn install
        cd ..
        ```

## Running the Application

1.  **Start Backend:**
    * *(Ensure venv is active)*
    * ```bash
        python app.py
        ```
    * *(Backend runs on http://127.0.0.1:5000)*

2.  **Start Frontend:**
    * *(Open a new terminal)*
    * ```bash
        cd frontend
        npm start
        # or: yarn start
        ```
    * *(App opens in browser, usually http://localhost:3000)*

## Notes

* **Model Download:** Downloading the `fine_tuned_bert` model is highly recommended. If missing, the backend will attempt to fine-tune a new one on startup, which is resource-intensive.
* **Coordinates:** Locations are approximate, based on city matching and random offsets.
* **Encryption:** AES key is ephemeral (changes on each server start), mainly for demonstration.
