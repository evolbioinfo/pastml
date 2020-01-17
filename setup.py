import os
from setuptools import setup, find_packages

setup(
    name='pastml',
    packages=find_packages(),
    include_package_data=True,
    package_data={'pastml': [os.path.join('templates', '*.html'), os.path.join('templates', '*.js'),
                             os.path.join('templates', 'js', '*.js'), os.path.join('templates', 'css', '*.css'),
                             os.path.join('templates', 'fonts', '*'),
                             os.path.join('..', 'README.md')]},
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='1.9.23',
    description='Ancestral character reconstruction and visualisation for rooted phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/pastml',
    keywords=['PASTML', 'visualisation', 'phylogeny', 'ancestral character reconstruction'],
    install_requires=['ete3', 'pandas', 'numpy', 'jinja2', 'scipy', 'itolapi', 'biopython'],
    entry_points={
            'console_scripts': [
                'pastml = pastml.acr:main',
                'geomap = pastml.visualisation.generate_geomap:main'
            ]
    },
)
