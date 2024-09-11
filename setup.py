import os
from setuptools import setup, find_packages

setup(
    name='pastml',
    packages=find_packages(),
    include_package_data=True,
    package_data={'pastml': [os.path.join('templates', '*.html'), os.path.join('templates', '*.js'),
                             os.path.join('templates', 'js', '*.js'), os.path.join('templates', 'css', '*.css'),
                             os.path.join('templates', 'fonts', '*'),
                             os.path.join('..', 'README.md'), os.path.join('..', 'LICENSE')]},
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    version='1.9.47',
    description='Ancestral character reconstruction and visualisation for rooted phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/pastml',
    keywords=['PASTML', 'visualisation', 'phylogeny', 'ancestral character reconstruction'],
    python_requires='>=3.10',
    install_requires=['ete3>=3.1.1', 'pandas>=1.0.0', 'numpy>=1.22', 'jinja2>=2.11.0', 'scipy==1.14.0', 'itolapi>=4.0.0', 'biopython>=1.70'],
    entry_points={
            'console_scripts': [
                'pastml = pastml.acr:main',
                'geomap = pastml.visualisation.generate_geomap:main',
                'transition_counter = pastml.utilities.transition_counter:main',
                'name_tree = pastml.tree:main'
            ]
    },
)
