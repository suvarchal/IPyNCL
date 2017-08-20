# IPyNCL: An IPython Notebook for NCAR Command Language

## Installation:

### I Have Jupyter and NCL already:

Get the Jupyter package either by downloading zip file from this git or clone this git repository.

Go into the package directory and install package by 

`python setup.py install`

Then open a jupyter notebook `jupyter notebook` and find NCL in list of notebooks to use.
Start scripting like in NCL. 

The NCL kernel for Jupyter will use what ever ncl version is currently sourced.

### I Dont Have Jupter Notebook:
First get Miniconda, it cleanly maintains different python versions/programs within their 
own virtual environments, to minimize any conflicts with other packages.
Moreover it is easy local installation, it sets the paths for you in the end. 
You can get the version for your operating system here http://conda.pydata.org/miniconda.html


Install jupyter and ncl by  

```conda create -n ncl_notebook  -c conda-forge ncl=6.4.0 jupyter```  

if you're on mac try the below instead

```conda create -n ncl_notebook  -c conda-forge -c ncar ncl=6.4.0 gsl jupyter```

Activate this environment with ```source activate ncl_notebook``` and launch the jupyter server with 
``` jupyter notebook```

