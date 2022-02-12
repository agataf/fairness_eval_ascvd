from setuptools import setup, find_packages

setup(
    name='fairness_ascvd',
    version='0.1.0',
    author='Agata Foryciarz and Stephen Pfohl',
    author_email='agataf@stanford.edu',
    url='https://github.com/agataf/fairness_ascvd',
    packages=find_packages(include=['prediction_utils', 'prediction_utils.*']),
    install_requires=['ConfigArgParse==1.2.3',
                      'GitPython==3.1.26',
                      'joblib==0.17.0',
                      'lifelines==0.25.7',
                      'matplotlib==3.3.2',
                      'numpy==1.19.2',
                      'pandas==1.1.3',
                      'pyarrow',
                      'PyYAML==6.0',
                      'scikit_learn==1.0.2',
                      'scipy==1.8.0',
                      'setuptools==58.0.4',
                      'seaborn==0.11.0',
                      'torch==1.8.0'
                     ]
)
