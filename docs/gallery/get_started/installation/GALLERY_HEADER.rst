Installation
==============

The recommended environment for running **RoboOrchardCore** is **Ubuntu 22.04** with
**Python 3.10**, which is the environment in which the code has been developed and
tested.


From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**RoboOrchardCore** packages are available on PyPI. The base install is the
slim configuration and CLI profile. Install the ``robotics`` extra when you
need tensor datatypes, environments, cameras, controllers, or kinematics; use
``all`` for the complete feature set.

.. code-block:: bash

    pip install robo_orchard_core

    # Robotics runtime and tensor-backed datatypes.
    pip install "robo_orchard_core[robotics]"

    # All official extras.
    pip install "robo_orchard_core[all]"


From Source
^^^^^^^^^^^^^^

For development purposes, you may want to install **RoboOrchardCore** from source. To install
**RoboOrchardCore** from source, you can clone the repository and install the packages using the
following command:

.. code-block:: bash

    # Clone the repository and move to the project directory first.

    make install-editable
