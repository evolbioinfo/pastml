from pastml.acr import pastml_pipeline
from pastml.models.GLMModel import GLM
from pastml.models.F81Model import F81
import os

FOLDER = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(FOLDER, 'data')
PREDICTORS_DIR = os.path.join(FOLDER, 'predictors')

pastml_pipeline(data=os.path.join(DATA_DIR, 'labelled_accessions_countries.txt'),
                data_sep=',',
                columns=['Country'],
                model=GLM,
                tree=os.path.join(DATA_DIR, 'timetree.nwk'),
                GLM_directory=PREDICTORS_DIR,
                html_compressed=os.path.join(DATA_DIR, 'RABV_canine_GLM5.bord.loc.2.map.html'),
                html=os.path.join(DATA_DIR, 'RABV_canine_GLM5.bord.loc.2.html'),
                verbose=True,
                threads=1)

#pastml_pipeline(data=os.path.join(DATA_DIR, 'labelled_accessions_countries.txt'),
#                data_sep=',',
#                columns=['Country'],
#                model=F81,
#                tree=os.path.join(DATA_DIR, 'timetree.nwk'),
#                html_compressed=os.path.join(DATA_DIR, 'RABV_canine.map.html'),
#                html=os.path.join(DATA_DIR, 'RABV_canine.html'),
#                verbose=True,
#                threads=1)
