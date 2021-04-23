from setuptools import setup, find_packages

setup(
    name='long-term-gdp-model',
    version='0.1.18',
    author='EA Data Team',
    description='GDP Model',
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    package_data={"": ["*"]},
    install_requires=[
        'numpy==1.19.0',
        'pandas',
        'datetime',
        'argparse',
        'openpyxl',
        'xlrd==1.2.0',
        'python-dotenv',
        'dbnomics',
        'pmdarima==1.8.2',
        'python-dateutil',
        'joblib',
        'statsmodels',
        'pathlib',
        'pytz',
        'requests'
    ]
)
