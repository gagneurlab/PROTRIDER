from setuptools import setup, find_packages

setup(
        name='outrider-prot',
        version='1.0',
        author="Ines Scheller, Daniela Klaproth-Andrade",
        author_email = "daniela.andrade@tum.de",
        description = 
        '''OUTRIDER-prot
            FIXME
        ''',
        python_requires = ">=3.7",
        install_requires = ['click',
                            'numpy',
                            'pandas',
                            'pyaml',
                            'pathlib',
                            'torch',
                            'tqdm',
                            'optht',
                            'pydeseq2',
                            'scipy',
                            'scikit-learn'
                            ],
        url='https://github.com/gagneurlab/outrider-prot',
        license='',
        packages=find_packages(include=['outrider-prot', 'outrider-prot.*']),
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'outrider-prot = outrider-prot.main:main',
            ],
        },
    include_package_data=True,
    #package_data={'outrider-prot': ['data/*.csv']},
)