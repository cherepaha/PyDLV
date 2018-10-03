# [PyDLV - decision landscape visualisation toolbox](https://github.com/cherepaha/PyDLV)
Decision landscapes are visually rich representations of decision making data. Any experimental data on mouse/hand tracking can potentially benefit from decision landscape visualisations (DLV).  

![alt text](http://rsos.royalsocietypublishing.org/content/royopensci/4/11/170482/F2.large.jpg "Decision landscape")

Dependencies
-------------
PyDLV is currently developed in Python 2.7 using the [numpy](https://sourceforge.net/projects/numpy/)/[scipy](https://github.com/scipy/scipy) framework, [pandas](https://github.com/pydata/pandas) is used for data manipulation. Another key dependencies are [matplotlib](https://github.com/matplotlib/matplotlib) and [seaborn](https://github.com/mwaskom/seaborn). The plans are to migrate to Python 3.6 by the end of 2018.

Installation
------------
To install, clone the repository and run

    python setup.py install

How to use
------------
To see DLV's in action, first download the data of O'Hora et al (2013) [here](http://doi.org/10.17605/OSF.IO/AHPV6). Then follow the [tutorial notebook](https://github.com/cherepaha/PyDLV/blob/master/demos/dlv_tutorial.ipynb). Note that tutorial and other demos are not installed along with the module, so you have to download them separately. Place the tutorial notebook somewhere along the downloaded data, and open the notebook. Adjust the path to the data file, and have fun! 

If you're not familiar with Jupyter Notebooks, here is a good [starting guide](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html). Also, [nbopen](https://github.com/takluyver/nbopen) is much recommended! 

When you're done with tutorial, you can use the scripts in the demos directory as examples. You may also find useful the notebook `demos/paper_figures.ipynb`, which generates almost all the figures from the paper (with the exception of Figure 3, which was generated manually in PowerPoint).

The project is currently in active development, stay tuned for updates!

How to cite
------------
The paper reporting the method is publised in Royal Society Open Science: http://rsos.royalsocietypublishing.org/content/4/11/170482

* "Decision landscapes: visualizing mouse-tracking data" A. Zgonnikov, A. Aleni, P. T. Piiroinen, D. O'Hora, M. di Bernardo, R. Soc. open sci. 2017 4 170482; DOI: 10.1098/rsos.170482, 8 November 2017 
