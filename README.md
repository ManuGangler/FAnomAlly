# FAnomAlly
Looking for Anomalies in Fink data ....

To get started, make sure to run the `setup.py` file to set up the necessary functions. You'll also need to download the `fink_utils` package.

Next, download the Fink data from: [Fink Portal](https://fink-portal.org/download). You can see an example with the data we used in our work here: [Example Data](https://drive.google.com/file/d/1VtJJIQdzevkUALEWU2fzrBzWVYPfeZmR/view?usp=sharing) (data1).

After that, you can perform data transfer or data reduction and feature extraction using the file `data-transfer_GP.py`. This process may take about half an hour for the (data1) file. Please note that this file fills the gaps with Gaussian Process regression. We also have a second file, `data-transfer.py`, which uses upper limits and weighted mean to fill gaps.

#### (Note: These two methods are direct applications on all objects. You can find explanations for each step on one object in the notebooks, the .ipynb versions.)

Simultaneously, go to `scripts/classification_functs/Run_jobs_th.py` and run the Python file. This step may take a significant amount of time, but it only needs to be done once after you download your data. It saves the classification of objects in the `Classifications_arch` folder so you don't have to call the server every time you want to see the classification.


Now, we start the analysis. We have two different files: one with optimization (`retrieve_data-exclud_matches.ipynb`) and one without optimization (`retrieve_data.ipynb`).


Here is a link to the six-month internship on this package, where we explain all the details, algorithms, and some results.
