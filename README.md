# Project Repo
## B-IT Pattern Recognition

Academic project, but with real world applications

Group members:

- Abdullah Abdullah
- Can Güney Aksakallı
- Kang Cifong
- Umut Hatipoğlu

***

## Folder structure
- `./data/` : folder for data; kept empty intentionally
    + please download the data and keep in this folder for discoverability
    + please do not commit or publish the data
- `./pattrex/` : source folder
    + contains source code for implemented modules
    + **Python 3.4**
    + **Numpy 1.9.6**
    + **Scipy 0.16.0**
- `./project-xx-demo-figures` : folder with output figures for demo of
  respective project
  <!-- may make into a single `demo` folder -->
- `convert-project-xx-demo.sh` : script to convert Jupyter Notebook for a
  project to slides
    + implemented/tested in **Jupyter** and **IPython 4.0.0**
- `project-xx-demo.ipynb` : notebook for demo of respective project
    + implemented/tested in **Jupyter** and **IPython 4.0.0**
    + to launch, after changing to this directory in a terminal, run ```jupyter-notebook```
- `project-xx-demo.slides.html` : slides generated from the demo notebooks for
  respective projects
    + just open in a web-browser. Tested in Chrome 47
    + **requires internet connection** to load properly
