# P300 Analysis from openvibe
**Convert - Plot ERPs - Run LDA - Check Features Map**
I use here 
- nice preprocessing tools from mne: Current Source Density (CSD),  Reference Electrode Standardization Technique (REST) @mne-tools [link](https://github.com/mne-tools/mne-python)
- Riemannian based artifact removal based Artifact Subspace Reconstruction from meegkit package (thanks@nbara) [link](https://github.com/nbara/python-meegkit) 
- Artifact channel and epoch rejection based on variance
 - Computes P300 ERP plots
 - Computes k-fold Linear Discriminant Analysis with shrinkage
 - Shows signed rsquared plots from wyrm (python port of bbci, thanks @bbci) [link](https://github.com/bbci/wyrm) 



| Signal output                                                | Signed r square output                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./img/8ch_output.png" alt="Analysis_P300" style="zoom:50%;" /> | <img src="./img/signed_rsquare.png" alt="Analysis_P300" style="zoom:60%;" /> |



# Requirements

- OpenViBE ( for converting to gdf)
- python 3.6+, MNE, jupyter

## Install python environment

- Download Anaconda Python
- open an anaconda prompt
- move to the current repository folder `cd %USERPROFILE%\Desktop\p300_analysis_from_openvibe`
- create the anaconda environment `conda env create -f environment.yml` 
- activate the environment `conda activate p300mne`
- install additional packages 
  - `pip install "git+https://github.com/nbara/python-meegkit"`
  - `pip install statsmodels pyriemann`
- Check whether mne was installed by pasting this code `python -c "import mne"`. It should trigger no error



## Convert OpenVibe .ov files into gdf files

Check my [tutorial here](https://github.com/lokinou/openvibe_to_gdf_tutorial)

## Accessing the notebook

- move to the current repository folder `cd %USERPROFILE%\Desktop\p300_analysis_from_openvibe`
- activate the environment `conda activate p300mne`
- execute the notebook: `jupyter lab p300_analysis.ipynb`
  - if jupyter lab crashes (win32api error), reinstall it from conda `conda install pywin32 jupyterlab`
  - if jupyter lab does not want to work, use jupyter notebook instead by executing `jupyter notebook p300mne.ipynb`

