Pipeline for producing scintillation data products from folded pulsar archives from the MeerKAT Thousand Pulsar Array.  Built to work on Ozstar supercomputer for parallel jobs on ~1000 sources.

Pipeline used and described in Main et al. 2022: https://doi.org/10.1093/mnras/stac3149

MeerKAT dynamic spectra (raw and filtered) are included on zenodo: https://doi.org/10.5281/zenodo.7261413

Exact modules used are listed in load_scintmodules.list

The main code is ScintPipeline.py, taking either a pulsar archive or filtered dynamic spectrum.  From an archive, it creates a 'raw' dynamic spectrum which has gaps from RFI, and includes intrinsic flux variations from the pulsar.  The dynamic spectrum is inpainted using a Wiener filter, which also attempts to remove the "window function" which is the product of the binary RFI mask and intrinsic flux variations. From the filtered dynamic spectrum, the script produces simple diagnostic plots of the dynamic spectrum, secondary spectrum, and autocorrelation functions.

Work so far has focussed on the secondary spectra - ACF fitting is currently a work in progress, and should be treated with caution.

Documentation will be added over time, but feel free to send queries to Robert Main (ramain at mpifr-bonn.mpg.de).