# Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning

Official implementation of:

**Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning**

Frank Hu, Michael S. Chen, Grant M. Rotskoff, Matthew W. Kanan, Thomas E. Markland

https://arxiv.org/abs/2408.08284

**Abstract:** Rapid determination of molecular structures can greatly accelerate workflows across many chemical disciplines. However, elucidating structure using only one-dimensional (1D) NMR spectra, the most readily accessible data, remains an extremely challenging problem because of the combinatorial explosion of the number of possible molecules as the number of constituent atoms is increased. Here, we introduce a multitask machine learning framework that predicts the molecular structure (formula and connectivity) of an unknown compound solely based on its 1D 1H and/or 13C NMR spectra. First, we show how a transformer architecture can be constructed to efficiently solve the task, traditionally performed by chemists, of assembling large numbers of molecular fragments into molecular structures. Integrating this capability with a convolutional neural network (CNN), we build an end-to-end model for predicting structure from spectra that is fast and accurate. We demonstrate the effectiveness of this framework on molecules with up to 19 heavy (non-hydrogen) atoms, a size for which there are trillions of possible structures. Without relying on any prior chemical knowledge such as the molecular formula, we show that our approach predicts the exact molecule 69.6% of the time within the first 15 predictions, reducing the search space by up to 11 orders of magnitude.

Data available on Zenodo at: https://zenodo.org/records/13892026
