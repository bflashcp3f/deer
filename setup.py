from setuptools import setup, find_packages

setup(
    name='deer',
    version="0.1.0",
    author='Fan Bai',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.12',
    install_requires=[
        'setuptools',
        'numpy>=1.24.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'sentence-transformers>=2.2.0',
        'openai>=1.0.0',
        'together>=1.0.0',
        'nltk>=3.8.0',
        'rank-bm25>=0.2.2',
        'tqdm>=4.65.0',
        'pandas>=2.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
    ],
    include_package_data=True,
)
