from pastml.acr import pastml_pipeline
from pastml.models.GLMModel import GLM
from pastml.models.F81Model import F81
import os

FOLDER = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(FOLDER, 'data_14')
PREDICTORS_DIR = os.path.join(FOLDER, 'predictors_14')

pastml_pipeline(data=os.path.join(DATA_DIR, 'data_14_named.txt'),
                data_sep=',',
                columns=['location14'],
                model=GLM,
                tree=os.path.join(DATA_DIR, 'H3N2.nwk'),
                GLM_directory=PREDICTORS_DIR,
                html_compressed=os.path.join(DATA_DIR, 'H3N2_GLM.map.html'),
                html=os.path.join(DATA_DIR, 'H3N2_GLM.html'),
                verbose=True,
                threads=1)

pastml_pipeline(data=os.path.join(DATA_DIR, 'data_14_named.txt'),
                data_sep=',',
                columns=['location14'],
                model=F81,
                tree=os.path.join(DATA_DIR, 'H3N2.nwk'),
                html_compressed=os.path.join(DATA_DIR, 'H3N2.map.html'),
                html=os.path.join(DATA_DIR, 'H3N2.html'),
                verbose=True,
                threads=1)
