# Multiview-CNN-Fusion-IACT
Here you can find the code and model visualisations for the research note "Multi-View Deep Learning for Imaging Atmospheric Cherenkov Telescopes" by Warnhofer H., Spencer S.T. and Mitchell, A.M.W.

For an overview about the model architectures and visualisations of the fusion methods used see: [HESS_CNN_Model_Architectures.pdf](https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/files/14356915/HESS_CNN_Model_Architectures.pdf)

## Guide
This code is used for building and evaluating multi-view CNNs for a binary event classification based on different approaches of fusing the information from four different views. A prerequesite for this code is the `dl1-data-handler` (see: [cta-observatory.dl1-data-handler](https://github.com/cta-observatory/dl1-data-handler)) with the respective camera geometry files that are needed for the image mapping of HESS-I data being available in 'Image Mapping Geometry Files'.

The code loads simulated HESS-I data and maps them to a 41x41 array using axial addressing. Two different types of events are available: Gamma-Ray events (`gamma`) and Cosmic Ray events (`proton`). For each event four images are available. The images get labeled and split up into training and testing data for multi-view and single-view CNNs. Dependend on the specified fusion type, a multi-view CNN is trained and evaluated. An ROC-curve is plotted and the data saved. The performance over the images size (as an estimate for the event intensity) is evaluated and saved in three different binnings. The ROC and the PerformanceOverTotalSize data from different runs can be combined using the plotting notebook. 

Example usage (see the top of `HESS_CNN_Run.py` for detailed information about the arguments that can be set when running the file):

EarlyMax Fusion, 50 epochs, 100000 events: `python HESS_CNN_Run.py -ft 'earlymax'`
LateFC Fusion, 250 epochs, 250000 events: `python HESS_CNN_Run.py -e 250 -ne 250000 -ft 'latefc' `

The following fusion types can be set:
- Early Max Position 1: `earlymax2`
- Early Conv Position 1: `earlyconv2`
- Early Concat Position 1: `earlyconcat2`
- Early Max Position 2: `earlymax`
- Early Conv Position 2: `earlyconv`
- Early Concat Position 2: `earlyconcat`
- Late FC: `latefc`
- Late Max: `latemax`
- Score Mean: `scoresum`
- Score Prod: `scoreproduct`
- Score Max: `scoremax`
