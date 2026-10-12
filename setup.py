# Based on princeton-nlp/LLM-Shearing (MIT License), Copyright (c) 2023 Princeton Natural Language Processing
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

import io
from setuptools import setup, find_packages 
import pathlib
import pkg_resources

with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name='elasticpruner',
    packages=["elasticpruner"],
    version='0.1',
    description='Elastic Pruner',
    author='dodowon',
    url='https://github.com/dodowon/ElasticPruner',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)

# auto-gptq cannot be installed
