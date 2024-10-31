import os
from setuptools import setup, find_packages

# Read the contents of README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''
    
# Read the contents of requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return [line.strip() for line in req.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='cna-modules',
    packages=find_packages(exclude=['tests*']),
    version='1.0.0',
    license='MIT',
    description='Cluster-Normalizes-Activate Modules',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Arseny Skryagin, Felix Divo, Mohammad Amin Ali',
    author_email='arseny.skryagin@cs.tu-darmstadt.de, felix.divo@cs.tu-darmstadt.de, amin.ali@stud.tu-darmstadt.de',
    url='https://github.com/ml-research/cna_modules',
    keywords=['Graph Neural Networks', 'Oversmoothing', 'CNA'],
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)