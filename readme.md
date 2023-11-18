
Para preprocesar
python .\main.py -i train.csv -d preprocess -s open_response -o we.joblib -p we

Para clasificar
python .\main.py -i train.csv,bow.joblib -d classify -s gs_text34 -o modelo.joblib