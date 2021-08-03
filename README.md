# image-processing
This pipeline is for counting RNA molecules in Drosophila NMJ. The input is a set of images, each of which contains the following channels:
    * hrp stained with DAPI
    * mCD8-GFP
    * shot FISH

The general idea is to identify and locate the NMJ, and then to count the RNA spots within its area. Because this is difficult to do with the membrane-fused GFP alone, its intersection with the DAPI signal is used instead. This works well because the latter is strongest precisely at the NMJ. More comments about the different steps in thet pipeline are present in the script file.

The outputs are stored in a separate directory and include a CSV file with the RNA molecule counts, as well as intermediate images for each of the important steps of the pipeline. These intermediate images have the filename of the original image, the one from which they were derived, appended with an extra identifier.

Configuration is done with a Python file called `process_config.py` and stored in the same directory as the script. See the included example file for the required parameters.
