from setuptools import setup, find_packages

setup(
    name='rcwa4d',
    version='0.0.0',
    # url='https://github.com/mypackage.git',
    author='Beicheng Lou',
    author_email='lbc45123@hotmail.com',
    description='RCWA for multilayered structure with incommensurate periodicities',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1'],
)