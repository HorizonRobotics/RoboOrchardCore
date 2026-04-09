System Overview
================

This section is the narrative starting point for ``robo_orchard_core``.
Read it when you want to understand the package from the top down before
diving into task-specific tutorials or symbol-level APIs.

Tutorial Coverage
-----------------

- :doc:`Best Practices for Config </build/gallery/tutorials/system_overview/nonb-01_config_design>` explains how runtime objects are described, validated, and serialized.
- :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` covers the local runtime loop and manager boundaries.
- :doc:`Remote Execution with Ray </build/gallery/tutorials/system_overview/nonb-03_remote_execution>` shows how those same interfaces scale across actor boundaries.
- :doc:`Controllers, Kinematics, and Motion Commands </build/gallery/tutorials/system_overview/nonb-04_control_and_kinematics>` maps motion goals to control and robot-structure layers.
- :doc:`CLI Tools and Extensions </build/gallery/tutorials/system_overview/nonb-05_tools_and_extensions>` explains operator-facing entry points and plugin extension points.

Whole To Part
-------------

RoboOrchardCore keeps a small set of package-level ideas stable across
training, simulation, and deployment code:

- **Configuration first**: ``utils.config`` uses typed config models and class
  factories so runtime construction stays explicit, serializable, and
  IDE-friendly.
- **Shared data contracts**: ``datatypes`` and math helpers keep transforms,
  cameras, joints, and frame graphs aligned across module boundaries.
- **Runtime orchestration**: ``envs``, ``policy``, ``controllers``, and
  ``kinematics`` define how agents, environments, and control logic interact.
- **Integration boundaries**: ``devices``, ``tools``, ``viz``, and remote
  wrappers attach the common runtime model to operators, notebooks, and
  external systems.

Module Walkthrough
------------------

.. list-table::
   :header-rows: 1

   * - Area
     - What it owns
     - Where to continue
   * - Configuration
     - Typed config schemas, validation, serialization, and class factories.
     - :doc:`Best Practices for Config </build/gallery/tutorials/system_overview/nonb-01_config_design>`
   * - Data model
     - Batch-first transforms, camera payloads, joint state, and frame-graph
       containers shared across the package.
     - :doc:`Data Types tutorials </build/gallery/tutorials/datatypes/index>`
   * - Environments and managers
     - Step/reset lifecycle, action and observation handling, event routing,
       and manager-term composition.
     - :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>`
   * - Remote execution
     - Ray-backed wrappers that keep the local environment and policy
       interfaces intact while moving execution behind actor boundaries.
     - :doc:`Remote Execution with Ray </build/gallery/tutorials/system_overview/nonb-03_remote_execution>`
   * - Control and kinematics
     - Robot-structure modeling, inverse-kinematics controllers, and the
       boundary between end-effector goals and environment actions.
     - :doc:`Controllers, Kinematics, and Motion Commands </build/gallery/tutorials/system_overview/nonb-04_control_and_kinematics>`
   * - Operator-facing tools
     - CLI commands, service helpers, and plugin-style extensions layered on
       top of the reusable runtime model.
     - :doc:`CLI Tools and Extensions </build/gallery/tutorials/system_overview/nonb-05_tools_and_extensions>`
   * - Policies, controllers, and kinematics
     - Lower-level API details for runtime decision and control abstractions
       after the overview tutorials establish the mental model.
     - :doc:`API reference </autoapi/index>`
   * - Integration utilities
     - Device adapters, notebook visualization, and other integration helpers
       that sit on top of the core contracts.
     - :doc:`API reference </autoapi/index>`

Suggested Reading Order
-----------------------

1. Start with :doc:`Best Practices for Config </build/gallery/tutorials/system_overview/nonb-01_config_design>` to understand how runtime objects are described and validated.
2. Continue with :doc:`Environment and Managers </build/gallery/tutorials/system_overview/nonb-02_env_and_managers>` for the local runtime loop.
3. Read :doc:`Remote Execution with Ray </build/gallery/tutorials/system_overview/nonb-03_remote_execution>` when your local abstractions need to scale across actor boundaries.
4. Move to :doc:`Controllers, Kinematics, and Motion Commands </build/gallery/tutorials/system_overview/nonb-04_control_and_kinematics>` to see how motion targets become executable robot commands.
5. Read :doc:`CLI Tools and Extensions </build/gallery/tutorials/system_overview/nonb-05_tools_and_extensions>` for operator-facing entry points and plugin packaging.
6. Use the :doc:`Data Types tutorials </build/gallery/tutorials/datatypes/index>` when you want deeper detail on the shared tensor-backed containers.
7. Use the :doc:`API reference </autoapi/index>` after the overview pages have established the mental model.

The tutorials in this section are intentionally architectural. They explain
how the main runtime pieces fit together. Data representation details live in
the separate :doc:`Data Types tutorials </build/gallery/tutorials/datatypes/index>`.
