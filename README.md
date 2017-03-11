# [PyDSV - decision landscape visualisation toolbox](http://www.ohoralab.com/)
Decision landscapes are visually rich representations of decision making data. Any experimental data on mouse/hand tracking can potentially benefit from decision landscape visualisations. We build the project up on [numpy](https://sourceforge.net/projects/numpy/)/[scipy](https://github.com/scipy/scipy) framework and use [pandas](https://github.com/pydata/pandas) to handle data manipulation. Another key dependencies are [matplotlib](https://github.com/matplotlib/matplotlib) and [seaborn](https://github.com/mwaskom/seaborn).

![alt text](https://github.com/cherepaha/PyDSV/blob/master/9276_9424.png "Example landscape visualisation")

fit_dl_to_data.py demonstrates how to fit decision landscape parameters to the mouse trajectory data and save the fitted parameters to the csv file. 

compare_dlv_two_subjects.py shows how to read the parameters from the csv file and visualise the decision lanscape in 3d using matplotlib.

To see the scripts in action, download the data of O'Hora et al (2013) [here](http://doi.org/10.17605/OSF.IO/AHPV6).

The project is currently in active development, stay tuned for updates!
