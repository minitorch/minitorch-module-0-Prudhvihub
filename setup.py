from setuptools import setup

setup(
    name="minitorch",
    version="0.1",
    packages=["minitorch"],
    install_requires=[
        "numpy",
        "pandas",
        "streamlit",
        "streamlit-ace",
        "pytest",
        "hypothesis",
        "typing_extensions",
        "pydot",
        "graphviz",
        "networkx",
    ],
)
