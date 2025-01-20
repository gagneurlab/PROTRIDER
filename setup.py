from setuptools import setup, find_packages

setup(
        name='outrider_prot',
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
        url='https://github.com/gagneurlab/outrider_prot',
        license='',
        packages=find_packages(include=['outrider_prot', 'outrider_prot.*']),
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'outrider_prot = outrider_prot.main:main',
            ],
        },
    include_package_data=True,
    #package_data={'outrider_prot': ['data/*.csv']},
)