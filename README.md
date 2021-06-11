



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
- create the anaconda environment `conda env create -f requirements.yaml` 
- activate the environment `conda activate p300mne`
- Check whether mne was installed by pasting this code `python -c "import mne"`. It should trigger no error



## Convert OpenVibe .ov files into gdf files

Check my [tutorial here](https://github.com/lokinou/openvibe_to_gdf_tutorial)

## Accessing the notebook

- move to the current repository folder `cd %USERPROFILE%\Desktop\p300_analysis_from_openvibe`
- activate the environment `conda activate p300mne`
- execute the notebook: `jupyter lab p300mne.ipynb`
- if jupyter lab crashes (win32api error), reinstall it from conda `conda install pywin32 jupyterlab`
- if jupyter lab does not want to work, use jupyter notebook instead by executing `jupyter notebook p300mne.ipynb`

