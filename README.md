# Multiview-CNN-Fusion-IACT
Here you can find the code and model visualisations for the research note "Multi-View Deep Learning for Imaging Atmospheric Cherenkov Telescopes" by Warnhofer H., Spencer S.T. and Mitchell, A.M.W.

For an overview about the model architectures and visualisations of the fusion methods used see: [HESS_CNN_Model_Architectures.pdf](https://github.com/hanneswarnhofer/multiview-cnn-fusion-iact/files/14356915/HESS_CNN_Model_Architectures.pdf)

## Guide
This code is used for building and evaluating multi-view CNNs for a binary event classification based on different approaches of fusing the information from four different views. A prerequesite for this code is the 'dl1-data-handler' (see: [cta-observatory.dl1-data-handler](https://github.com/cta-observatory/dl1-data-handler)) with the respective camera geometry files that are needed for the image mapping of HESS-I data being available in 'Image Mapping Geometry Files'.

