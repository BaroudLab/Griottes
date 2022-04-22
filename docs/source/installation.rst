Installation
============

From source
------------

Clone the repository from github

.. code-block:: console 

    git clone https://gitlab.pasteur.fr/BaroudLab/Griottes.git

and run installation

.. code-block:: console 

    cd Griottes
    pip install .

Otherwise, a single-line option will be:

.. code-block:: console 

    pip install git+https://gitlab.pasteur.fr/BaroudLab/Griottes.git

In order to use jupyter notebooks, you will need to install jupyter

.. code-block:: console 

    pip install jupyterlab
    jupyter lab

From Docker
-----------

Start with pulling the container

.. code-block:: console 

    docker run -it -p 8888:8888 ghcr.io/baroudlab/griottes:latest


This will open jupyter lab in the folder with the sample notebooks (/home/jovyan/Griottes/example_notebooks) also containing paper figures.

If you want to customize starting folder, just run

.. code-block:: console 

    docker run -it -p 8888:8888 ghcr.io/baroudlab/griottes:latest jupyter lab --notebook-dir /home/jovyan/

In order to provide your own data to the notebooks, bind your local folder as follows:

.. code-block:: console 

    docker run -it -p 8892:8888 -v "${PWD}":/home/jovyan/work ghcr.io/baroudlab/griottes:latest jupyter lab --notebook-dir /home/jovyan/work

From Binder
-----------

Try interactive notebooks in `Binder <https://mybinder.org/v2/gh/BaroudLab/Griottes.git/main>`_
