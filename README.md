# This repository serves as codespace for assigment.   
---
### Project structure   

```
DATA
 ┣ A composite ranking of risk factors for COVID-19 time-to-event data from a Turkish cohort.pdf
 ┣ A Multi-Machine Learning Consensus Model Based on Clinical Features Reveals That Interleukin-10 Derived from Monocytes Leads to a Poor Prognosis in Pa.pdf
 ┣ Coronavirus disease 2019 (COVID‑19) survival analysis using deep learning and Cox regression model.pdf
 ┣ Development and validation of a risk factor-based system to predict short-term survival in adult hospitalized patients with COVID-19 a multicenter, retrospective, cohort study.pdf
 ┗ Time to recovery of COVID-19 patients and its predictors a retrospective cohort study in HUCSH, Sidama, Ethiopia.pdf
src
 ┣ app.py
 ┣ constants.py
 ┣ extraction.py
 ┣ loaders.py
 ┣ preprocessing.py
 ┣ utils.py
 ┗ visualization.py
 .gitignore
 README.md
```
### Requirements   

```
streamlit
pandas
matplotlib
wordcloud
scikit-learn
yake
keybert
PyMuPDF
unidecode
```

### Setup instructions   

It is recommended to use a **virtual environment** to isolate dependencies.  
To create and activate one, open your terminal in the project folder and type:  

#### On Windows:  
```bash
python -m venv venv
```
To activate venv, if it already isn't, run in terminal:  
```bash
venv\Scripts\activate.ps1
```

#### On macOS / Linux:  
```bash
python3 -m venv venv
```
To activate venv, if it already isn't, run in terminal:  
```bash
source venv/bin/activate
```

### To install all required dependencies:  
Firstly, make sure you have **venv** activated in your terminal. You shold see **(venv)** infront of your path.
```bash
pip install -r requirements.txt
```

### Running the app  
```bash
streamlit run src/app.py
```
---
# About this project  

The app allows you to:  
- Upload text or PDF documents.  
- Preprocess the text automatically.  
- Extract keywords using one of three methods:  
    - TF-IDF
    - YAKE
    - KeyBERT

- Compare results across methods.  
- Visualize extracted keywords (bar chart, word cloud).  
- Export keyword results as CSV.  