from pastml.acr import pastml_pipeline
from pastml.models.GLMModel import GLM
from pastml.models.F81Model import F81
import os

FOLDER = os.path.abspath(os.path.dirname(__file__))
PREDICTORS_DIR = os.path.join(FOLDER, 'predictors')


# pastml_pipeline(data=os.path.join(FOLDER, 'batRABV_hostLocation.txt'),
#                 data_sep=',',
#                 columns=['host'],
#                 model=F81,
#                 tree=os.path.join(FOLDER, 'batRABV.fas.raxml.bestTreeCollapsed.result.date.nexus'),
#                 html_compressed=os.path.join(FOLDER, 'RABV.map.html'),
#                 html=os.path.join(FOLDER, 'RABV.html'),
#                 verbose=True,
#                 threads=1)

pastml_pipeline(data=os.path.join(FOLDER, 'batRABV_hostLocation.txt'),
                data_sep=',',
                columns=['host'],
                model=GLM,
                tree=os.path.join(FOLDER, 'batRABV.fas.raxml.bestTreeCollapsed.result.date.nexus'),
                # GLM_directory=PREDICTORS_DIR,
                html_compressed=os.path.join(FOLDER, 'RABV_GLM.map.html'),
                html=os.path.join(FOLDER, 'RABV_GLM.html'),
                parameters=os.path.join(FOLDER, 'params.GLM.tab'),
                verbose=True,
                threads=1)
