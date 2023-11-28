
### Para preprocesar
```bash
python .\main.py -i train.csv -d preprocess -s open_response -o <tecnica_vectorizacion>.joblib -p <tecnica_vectorizacion>
```
Ej para preprocesar con tfidf: 
```bash
python .\main.py -i train.csv -d preprocess -s open_response -o tfidf.joblib -p tfidf
```
### Para clasificar
```bash
python .\main.py -i train.csv,<tecnica_vectorizacion>.joblib -d classify -s gs_text34 -o modelo.joblib
```
Ej para clasificar usando documentos vectorizados con bow:
```bash
python .\main.py -i train.csv,bow.joblib -d classify -s gs_text34 -o modelo.joblib
```
bow.joblib deberá haberse generado antes mediante el preprocessor adecuado: 
```bash
python .\main.py -i train.csv -d preprocess -s open_response -o bow.joblib -p bow
```
### Para clustering
```bash
python main.py -i <tecnica_vectorizacion>.joblib -d clustering -s gs_text34 -p 12 -o clustering.joblib
```
Ej para hacer una tarea de clustering con documentos vectorizados con we: 
```bash
python main.py -i we.joblib -d clustering -s gs_text34 -p 12 -o clustering.joblib
```
we.joblib deberá haberse generado antes mediante el preprocessor adecuado:
```bash
python .\main.py -i train.csv -d preprocess -s open_response -o we.joblib -p we
```

## Para RQ2
```bash
python main.py -i bow.joblib,tfidf.joblib,we.joblib,transformers.joblib -d Q2 -s gs_text34 -p 2,15
```
## Para RQ1
```bash
python main.py -i train.csv -d doQ1
```
### <tecnica_vectorizacion> = bow | tfidf | we | transformers