
Para preprocesar
python .\main.py -i train.csv -d preprocess -s open_response -o we.joblib -p we

Para clasificar
python .\main.py -i train.csv,bow.joblib -d classify -s gs_text34 -o modelo.joblib

Para clustering

python main.py -i transformers.joblib -d clustering -s gs_text34 -p 12
