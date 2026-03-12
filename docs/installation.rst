Installation
============

Prerequisites
-------------

* Python 3.11 or higher
* pip

Install from source
-------------------

.. tabs::

   .. group-tab:: Windows

      .. code-block:: bash

         # Clone the repository
         git clone https://github.com/ffe-munich/CHAMPPy.git
         cd CHAMPPy

         # Create a virtual environment:
         py -m venv .venv

         # Activate virtual environment
         .\.venv\Scripts\activate

         # Install the package
         pip install .


   .. group-tab:: Linux/Mac

      .. code-block:: bash

         # Clone the repository
         git clone https://github.com/ffe-munich/CHAMPPy.git
         cd CHAMPPy

         # Create a virtual environment
         python -m venv .venv

         # Activate virtual environment
         source .venv/bin/activate

         # Install the package
         pip install .



Install from PyPI
-----------------

.. tabs::

   .. group-tab:: Windows

      .. code-block:: bash

         # Create a virtual environment
         py -m venv .venv

         # Activate virtual environment
         .\.venv\Scripts\activate

         pip install champpy

   .. group-tab:: Linux/Mac

      .. code-block:: bash

         # Create a virtual environment
         python -m venv .venv

         # Activate virtual environment
         source .venv/bin/activate

         pip install champpy