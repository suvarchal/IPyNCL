# IPyNCL: A IPython kernel for NCAR Command Language

##Installation:

###I Have IPython Notebook:

Get the IPyNCL package either by downloading zip file from this git or clone this git repository.

Go into the package directory and install package by 

`python setup.py install`

Then open a ipython notebook `ipython notebook` and find NCL in list of notebooks to use. Start scripting like in NCL. 

###I Dont Have IPython Notebook:
I recommend to get Miniconda, it cleanly maintains different python versions/programs within their own virtual environments, to minimize any conflicts with other packages. Moreover it is easy local installation, it sets the paths for you in the end. You can get the version for your operating system here http://conda.pydata.org/miniconda.html

Install ipython and notebook by 
`conda install ipython-notebook`

*If you plan to use more python programs in future, it is recommened to make a seperate environment for NCL 
`conda create -n NCL ipython-notebook` and activate your profile by `source activate NCL`*

Then read **I Have Ipython Notebook** instructions above.
