from setuptools import setup, find_packages

setup(
    name="P-Funk",
    version="0.3",
    description="Python package for Fresco UNcertainty FunKulations",
    author="Caleb Marshall",
    author_email="camarsha@unc.edu",
    url="https://github.com/camarsha/pfunk",
    packages=find_packages(),
    install_requires=[
        "pandas <= 24.0",
        "matplotlib > 3.0",
        "emcee>2.2.0",
        "dynesty",
        "zeus-mcmc",
    ],
)
