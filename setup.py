from setuptools import find_packages, setup

setup(
    name='mlwolf',
    packages=find_packages(),
    version='0.1.0',
    description='Object oriented api over mlflow tracking',
    author='',
    install_requires=[
        'mlflow',
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    license='MIT',
)
