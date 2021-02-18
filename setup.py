from setuptools import find_packages, setup

setup(
    name='mlflowstone',
    packages=find_packages(),
    version='0.1.1',
    license='MIT',
    description='Object oriented api over mlflow tracking',
    author='Luciano Lorenti',
    url='https://github.com/lucianolorenti/mlflowstone',
    install_requires=[
        'mlflow',
        'numpy',
        'pandas',
        'scikit-learn'
    ],

)
