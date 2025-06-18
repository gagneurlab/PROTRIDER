from setuptools import setup, find_packages

setup(
        name='protrider',
        version='1.0',
        author="Ines Scheller, Daniela Klaproth-Andrade",
        author_email = "daniela.andrade@tum.de",
        description = 
        '''PROTRIDER
            Protein outlier detection method
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
                            'scikit-learn',
                            'pathlib'
                            ],
        url='https://github.com/gagneurlab/PROTRIDER',
        license='',
        packages=find_packages(include=['protrider',
                                        'protrider.*']),
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'protrider = protrider.main:main',
            ],
        },
    include_package_data=True,
    #package_data={'protrider': ['data/*.csv']},
)
