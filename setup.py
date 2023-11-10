from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSION = '0.2.4'
DESCRIPTION = 'Streamlining the process of multi-prompting LLMs with chains'

setup(
    name="flowchat",
    version=VERSION,
    author="Hinson Chan",
    author_email="<yhc3141@gmail.com>",
    maintainer="Hinson Chan",
    maintainer_email="<yhc3141@gmail.com>",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["openai"],
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license='MIT',
    keywords="openai gpt3 gpt-3 gpt4 gpt-4 chatbot ai nlp prompt prompt-engineering toolkit",
    url="https://github.com/flatypus/flowchat",
    project_urls={
        "Repository": "https://github.com/flatypus/flowchat",
        "Issues": "https://github.com/flatypus/flowchat/issues",
    },

)
