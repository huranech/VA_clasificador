from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict


def mapeo_a_numeros(y_labels):
    '''
    '''
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(y_labels)

    return y_labels


def minimalist(y_labels):
    '''
    '''
    mapeo_categorias = {
    'Diarrhea/Dysentery': 'Infecciones',
    'Other infectious diseases': 'Infecciones',
    'AIDS': 'Infecciones',
    'Sepsis': 'Infecciones',
    'Meningitis': 'Infecciones',
    'Meningitis/Sepsis': 'Infecciones',
    'Malaria': 'Infecciones',
    'Encephalitis': 'Infecciones',
    'Measles': 'Infecciones',
    'Hemorrhagic Fever': 'Infecciones',
    'TB': 'Infecciones',
    'Other Infectious Diseases': 'Infecciones',
    
    'Leukemia/Lymphomas': 'Neoplasmas',
    'Colorectal Cancer': 'Neoplasmas',
    'Lung Cancer': 'Neoplasmas',
    'Cervical Cancer': 'Neoplasmas',
    'Breast Cancer': 'Neoplasmas',
    'Stomach Cancer': 'Neoplasmas',
    'Prostate Cancer': 'Neoplasmas',
    'Esophageal Cancer': 'Neoplasmas',
    'Other Cancers': 'Neoplasmas',
    
    'Diabetes': 'Endocrinas',
    
    'Epilepsy': 'Nerviosas',
    
    'Stroke': 'Circulatorias',
    'Acute Myocardial Infarction': 'Circulatorias',
    'Other Cardiovascular Diseases': 'Circulatorias',
    
    'Pneumonia': 'Respiratorias',
    'Asthma': 'Respiratorias',
    'COPD': 'Respiratorias',
    
    'Cirrhosis': 'Digestivas',
    'Other Digestive Diseases': 'Digestivas',
    
    'Renal Failure': 'Genitourinarias',
    
    'Preterm Delivery': 'Embarazo y Parto',
    'Stillbirth': 'Embarazo y Parto',
    'Maternal': 'Embarazo y Parto',
    'Birth asphyxia': 'Embarazo y Parto',
    
    'Congenital malformation': 'Malformaciones Congénitas',
    
    'Bite of Venomous Animal': 'Lesiones y Envenenamientos',
    'Poisonings': 'Lesiones y Envenenamientos',
    
    'Road Traffic': 'Causas Externas',
    'Falls': 'Causas Externas',
    'Homicide': 'Causas Externas',
    'Fires': 'Causas Externas',
    'Drowning': 'Causas Externas',
    'Suicide': 'Causas Externas',
    'Violent Death': 'Causas Externas',
    'Other Injuries': 'Causas Externas',

    'Hemorrhagic fever': 'Otras Causas',
    'Other Defined Causes of Child Deaths': 'Otras Causas',
    'Other Non-communicable Diseases': 'Otras Causas'
    }
    y_labels_mapeadas = y_labels.map(mapeo_categorias)
    for i, item in enumerate(y_labels_mapeadas):
        if pd.isna(item):
            repetidas = y_labels[i]
            print(repetidas)
    return y_labels_mapeadas