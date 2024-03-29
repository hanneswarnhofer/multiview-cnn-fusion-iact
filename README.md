# Multiview-CNN-Fusion-IACT
Here you can find the code and model visualisations for the research note "Multi-View Deep Learning for Imaging Atmospheric Cherenkov Telescopes" by Warnhofer H., Spencer S.T. and Mitchell, A.M.W.

### For an overview about the model architectures and visualisations of the fusion methods used see: [HESS_CNN_Model_Architectures_Images.pdf](https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/blob/main/HESS_CNN_Model_Architectures_Images.pdf)


## Guide
This code is used for building and evaluating multi-view CNNs for a binary event classification based on different approaches of fusing the information from four different views. A prerequesite for this code is the `dl1-data-handler` (see: [cta-observatory.dl1-data-handler](https://github.com/cta-observatory/dl1-data-handler)) with the respective camera geometry files that are needed for the image mapping of HESS-I data available in this repository: `geometry2d3.json` and `HESS-I.camgeom.fits.gz`

The code loads simulated HESS-I data and maps each telescope to a 41x41 image using axial addressing. Two different types of events are available: Gamma-Ray events (`gamma`) and Cosmic Ray events (`proton`). For each event four images are available. The images get labeled and split up into training and testing data for multi-view and single-view CNNs. Dependend on the specified fusion type, a multi-view CNN is trained and evaluated. An ROC-curve is plotted and the data saved. The performance over the images size (as an estimate for the event intensity) is evaluated and saved in three different binnings. The ROC and the PerformanceOverTotalSize data from different runs can be combined using the `HESS_CNN_PlottingNotebook.ipynb`. 

Example usage (see the top of `HESS_CNN_Run.py` for detailed information about the arguments that can be set when running the file):

- EarlyMax Fusion, 50 epochs *(default)*, 100000 events *(default)*: `python HESS_CNN_Run.py -ft 'earlymax'`
- LateFC Fusion, 250 epochs, 250000 events: `python HESS_CNN_Run.py -e 250 -ne 250000 -ft 'latefc' `

The following fusion types can be set:
- Early Max Position 1: `earlymax`
- Early Conv Position 1: `earlyconv`
- Early Concat Position 1: `earlyconcat`
- Early Max Position 2: `earlymax2`
- Early Conv Position 2: `earlyconv2`
- Early Concat Position 2: `earlyconcat2`
- Late FC: `latefc`
- Late Max: `latemax`
- Score Mean: `scoremean`
- Score Prod: `scoreproduct`
- Score Max: `scoremax`

With any questions regarding this code, feel free to contact hannes.warnhofer@fau.de 

## Result Plots
In subjectively beautiful *(top)* and colorblind and b/w friendly *(bottom)*:
<img src="https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/blob/main/2024-02-21_FullPlot.png" width=70% height=70%>
<img src="https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/blob/main/2024-02-19_FullPlot_coloblindfriendly.png" width=70% height=70%>

Plot with results from both Early Fusion positions included:
<img src="https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/blob/main/2024-02-21_FullPlot_BothEarlyPositions.png" width=80% height=80%>

Zenodo DOI: https://doi.org/10.5281/zenodo.10868020
