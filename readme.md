
Para preprocesar
python .\main.py -i train.csv -d preprocess -s open_response -o tfidf.joblib -p tfidf

Para clasificar
python .\main.py -i train.csv,tfidf.joblib -d classify -s gs_text34 -o modelo.joblib

Para clustering
python main.py -i tfidf.joblib -d clustering -s gs_text34 -p 12 -o clustering.joblib

Para RQ2
python main.py -i bow.joblib -d Q2 -s gs_text34 -p 2,48 -o Q2.joblib