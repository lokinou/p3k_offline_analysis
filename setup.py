from setuptools import setup, find_packages

VERSION = '0.1.7'
DESCRIPTION = 'p3k - yet another P300 offline classification tool'
LONG_DESCRIPTION = 'P300 offline classification from BCI2000 P3Speller and OpenVibe P300 scenario'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="p3k",
    version=VERSION,
    author="Loic Botrel",
    author_email="<loic.botrel@uni-wuerzburg.de>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scikit-learn>=0.22', 'pandas>=1.3.2', 'seaborn>=0.11.2', 'numpy>=1.20.2', 'mne>=0.23', 'matplotlib', 'BCI2kReader'], #
	#'meegkit @ git+https://github.com/nbara/python-meegkit.git#egg=meegkit[extra]'],
	#dependency_links=['git+https://github.com/nbara/python-meegkit.git#egg=meegkit'],
	python_requires='>3.7',
    keywords=['python', 'bci2000', 'p300', 'speller', 'bci', 'bmi', 'eeg', 'openvibe', 'lda', 'mne'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
	license='MIT',
	url='https://github.com/lokinou/p300_analysis_from_openvibe_and_bci2000'
)