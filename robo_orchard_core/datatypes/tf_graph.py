# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Any, Generic, Sequence, TypeVar

from typing_extensions import Self

from robo_orchard_core.datatypes.dataclass import DataClass
from robo_orchard_core.datatypes.geometry import BatchFrameTransform

EDGE_TYPE = TypeVar("EDGE_TYPE")
NODE_TYPE = TypeVar("NODE_TYPE")


__all__ = [
    "BatchFrameTransformGraphState",
    "BatchFrameTransformGraph",
]


class EdgeGraph(Generic[EDGE_TYPE, NODE_TYPE]):
    """A generic edge graph data structure.

    Template Parameters:
        EDGE_TYPE: The type of the edges in the graph.
        NODE_TYPE: The type of the nodes in the graph.
    """

    edges: dict[str, dict[str, EDGE_TYPE]]
    """Graph is represented as a set of edges."""
    nodes: dict[str, NODE_TYPE]
    """The nodes are represented as string dict."""

    def __init__(self):
        self.edges = {}
        self.nodes = {}
        self._in_degree = {node_id: 0 for node_id in self.nodes}

    def _add_node(self, node_id: str, node: NODE_TYPE):
        """Add a node to the graph.

        Raises:
            ValueError: If the node already exists.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists.")
        self.nodes[node_id] = node
        self.edges[node_id] = {}
        self._in_degree[node_id] = 0

    def _add_edge(self, from_node: str, to_node: str, edge: EDGE_TYPE):
        """Add an edge between two nodes.

        Raises:
            ValueError: If either node does not exist or if the edge already
                exists.
        """
        if from_node not in self.nodes:
            raise ValueError(f"From node {from_node} does not exist.")
        if to_node not in self.nodes:
            raise ValueError(f"To node {to_node} does not exist.")
        if to_node in self.edges[from_node]:
            raise ValueError(
                f"Edge from {from_node} to {to_node} already exists."
            )
        self.edges[from_node][to_node] = edge
        self._in_degree[to_node] += 1

    def connected_subgraph_number(self) -> int:
        """Count the number of all subgraphs in the graph."""

        zero_in_degree_nodes = [
            node_id
            for node_id, degree in self._in_degree.items()
            if degree == 0
        ]
        return max(1, len(zero_in_degree_nodes))

    def get_path_by_bfs(
        self, src_node_id: str, dst_node_id: str
    ) -> list[EDGE_TYPE] | None:
        """Get the path from src_node_id to dst_node_id.

        This method uses breadth-first search (BFS) to find the shortest path
        between two nodes in the graph. If no path exists, it returns None.

        Args:
            src_node_id (str): The ID of the source node.
            dst_node_id (str): The ID of the destination node.

        Returns:
            list[EDGE_TYPE] | None: A list of edges representing the path from
            src_node_id to dst_node_id.

        """
        if (
            src_node_id not in self.nodes
            or dst_node_id not in self.nodes
            or src_node_id == dst_node_id
        ):
            return None

        # apply breadth-first search (BFS) to find the path
        queue = [src_node_id]
        visited = {src_node_id}
        # Record who first visited the node.
        # This is used to reconstruct the shortest path.
        parent_map: dict[str, str | None] = {src_node_id: None}
        while queue:
            current_node = queue.pop(0)
            if current_node == dst_node_id:
                break
            for neighbor in self.edges[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current_node
                    queue.append(neighbor)
                else:
                    continue

        if dst_node_id not in visited:
            return None

        # reconstruct the path
        path = []
        current_node = dst_node_id
        while current_node is not None:
            parent = parent_map[current_node]
            if parent is not None:
                edge = self.edges[parent].get(current_node)
                if edge is not None:
                    path.append(edge)
            current_node = parent
        path.reverse()
        return path


class BatchFrameTransformGraphState(DataClass):
    tf_list: list[BatchFrameTransform]
    bidirectional: bool = True
    static_tf: list[bool] | None = None

    def __post_init__(self):
        if self.static_tf is not None and len(self.static_tf) != len(
            self.tf_list
        ):
            raise ValueError(
                "static_tf and tf_list must have the same length."
            )


class BatchFrameTransformGraph(EdgeGraph[BatchFrameTransform, str]):
    """A graph structure for batch frame transforms.

    This graph structure is specifically designed to handle batch frame
    transforms, where each edge represents a transformation between two frames.
    The nodes are identified by their frame IDs.

    The graph exposes an effective batch dimension. Each non-mirrored edge
    must have batch size ``1`` or the graph effective batch size ``N``.
    Batch-size-1 edges are treated as singleton edges that can broadcast
    during transform composition and can be materialized explicitly with
    :meth:`repeat_singleton_tfs`. Empty sliced graphs are represented with
    batch size ``0``.

    Args:
        tf_list (list[BatchFrameTransform] | None): A list of
            BatchFrameTransform objects to initialize the graph with.
            If None, an empty graph is created.
        bidirectional (bool): Whether to add mirrored edges in the opposite
            direction. Defaults to True.
        static_tf (list[bool] | None): A list of booleans indicating whether
            each BatchFrameTransform is static. If None, all transforms are
            considered non-static. Defaults to None.

    """

    edges: dict[str, dict[str, BatchFrameTransform]]
    """Graph is represented as a set of tf.

    Mirrored(inversed) edges are also stored in the graph.
    """

    def __init__(
        self,
        tf_list: list[BatchFrameTransform] | None,
        bidirectional: bool = True,
        static_tf: list[bool] | None = None,
    ):
        super().__init__()
        self._bidirectional = bidirectional
        # the dict to store mirrored edges
        self._mirrored_edges: dict[str, dict[str, BatchFrameTransform]] = {}
        # the dict to store whether the edge is static (include mirrored edges)
        self._static_edges: dict[str, dict[str, bool]] = {}

        if tf_list is not None:
            self._validate_tf_batch_contract(tf_list)
            self._add_tf(
                tf_list, bidirectional=bidirectional, static_tf=static_tf
            )

    def __repr__(self) -> str:
        return (
            f"BatchFrameTransformGraph(nodes={self.nodes.keys()}, "
            f"num_edges={sum(len(v) for v in self.edges.values())})"
        )

    @staticmethod
    def _validate_tf_batch_contract(
        tf_list: Sequence[BatchFrameTransform],
    ) -> int:
        """Validate the effective batch contract for non-mirrored edges.

        Args:
            tf_list (Sequence[BatchFrameTransform]): The non-mirrored edge
                transforms to validate.

        Returns:
            int: The effective graph batch size.

        Raises:
            ValueError: If any edge batch size is incompatible with the graph
                effective batch size.
        """
        if len(tf_list) == 0:
            return 0

        batch_sizes = [tf.batch_size for tf in tf_list]
        effective_batch_size = max(batch_sizes)
        allowed_batch_sizes = {effective_batch_size}
        if effective_batch_size > 0:
            allowed_batch_sizes.add(1)

        invalid_batch_sizes = sorted(
            {
                batch_size
                for batch_size in batch_sizes
                if batch_size not in allowed_batch_sizes
            }
        )
        if invalid_batch_sizes:
            raise ValueError(
                "All BatchFrameTransformGraph edges must have batch size 1 "
                f"or {effective_batch_size}. Got invalid batch sizes "
                f"{invalid_batch_sizes}."
            )
        return effective_batch_size

    def _non_mirrored_tf_list(self) -> list[BatchFrameTransform]:
        return self.as_state().tf_list

    @property
    def batch_size(self) -> int:
        """Get the effective graph batch size.

        Each non-mirrored edge must have batch size ``1`` or the returned
        effective batch size ``N``. Empty graphs and empty slices return
        batch size ``0``.

        Returns:
            int: The effective graph batch size.
        """
        return self._validate_tf_batch_contract(self._non_mirrored_tf_list())

    @staticmethod
    def _slice_batch_size(
        idx: list[int] | slice | int, batch_size: int
    ) -> int:
        """Resolve the resulting batch size for a graph slice."""
        batch_indices = list(range(batch_size))
        if isinstance(idx, int):
            _ = batch_indices[idx]
            return 1
        if isinstance(idx, slice):
            return len(batch_indices[idx])
        return len([batch_indices[i] for i in idx])

    @staticmethod
    def _repeat_singleton_tf(
        tf: BatchFrameTransform, batch_size: int
    ) -> BatchFrameTransform:
        repeated_timestamps = None
        if tf.timestamps is not None:
            repeated_timestamps = tf.timestamps * batch_size
        return tf.repeat(batch_size=batch_size, timestamps=repeated_timestamps)

    def repeat_singleton_tfs(self, batch_size: int | None = None) -> Self:
        """Repeat singleton non-mirrored edges to a target batch size.

        Args:
            batch_size (int | None, optional): The target batch size for
                singleton edges. If None, use the graph effective batch size.
                Defaults to None.

        Returns:
            Self: A new graph whose singleton non-mirrored edges are repeated
            to ``batch_size``.

        Raises:
            ValueError: If ``batch_size`` is negative or incompatible with any
                non-singleton edge already stored in the graph.
        """
        target_batch_size = (
            self.batch_size if batch_size is None else batch_size
        )
        if target_batch_size < 0:
            raise ValueError("batch_size must be non-negative.")

        state = self.as_state()
        repeated_tf_list = []
        for tf in state.tf_list:
            if tf.batch_size == 1:
                if target_batch_size == 0:
                    repeated_tf_list.append(tf[:0])
                elif target_batch_size == 1:
                    repeated_tf_list.append(tf)
                else:
                    repeated_tf_list.append(
                        self._repeat_singleton_tf(
                            tf, batch_size=target_batch_size
                        )
                    )
            elif tf.batch_size == target_batch_size:
                repeated_tf_list.append(tf)
            else:
                raise ValueError(
                    "Cannot repeat singleton graph edges to batch size "
                    f"{target_batch_size} because a non-singleton edge has "
                    f"batch size {tf.batch_size}."
                )

        return type(self)(
            tf_list=repeated_tf_list,
            bidirectional=state.bidirectional,
            static_tf=state.static_tf,
        )

    def is_mirrored_tf(
        self, parent_frame_id: str, child_frame_id: str
    ) -> bool:
        """Check if the edge is a mirrored edge."""
        return (
            parent_frame_id in self._mirrored_edges
            and child_frame_id in self._mirrored_edges[parent_frame_id]
        )

    def is_static_tf(self, parent_frame_id: str, child_frame_id: str) -> bool:
        """Check if the edge is a static edge."""
        return parent_frame_id in self._static_edges and self._static_edges[
            parent_frame_id
        ].get(child_frame_id, False)

    def frame_names(self) -> set[str]:
        """Get all frame names in the graph."""
        return set(self.nodes.keys())

    def _add_node(self, node_id: str, node: str):
        """Add a node to the graph.

        Overwrites the base class method to ensure that mirrored edges
        and static edges are initialized correctly.

        Raises:
            ValueError: If the node already exists.

        """
        if node_id in self._mirrored_edges:
            raise ValueError(
                f"Node {node_id} already exists in mirrored edges."
            )
        if node_id in self._static_edges:
            raise ValueError(f"Node {node_id} already exists in static edges.")

        ret = super()._add_node(node_id, node)

        self._mirrored_edges[node_id] = {}
        self._static_edges[node_id] = {}
        return ret

    def _add_edge(
        self,
        from_node: str,
        to_node: str,
        edge: BatchFrameTransform,
        bidirectional: bool = True,
        is_static: bool = False,
    ):
        """Add an edge between two nodes.

        Overwrites the base class method to handle mirrored edges
        and static edges.
        """

        super()._add_edge(from_node, to_node, edge)
        if is_static:
            self._static_edges[from_node][to_node] = True
        if bidirectional:
            # Add the mirrored edge in the opposite direction
            mirrored_tf = edge.inverse()
            mirrored_from_node = mirrored_tf.parent_frame_id
            mirrored_to_node = mirrored_tf.child_frame_id
            self._mirrored_edges[mirrored_from_node][mirrored_to_node] = (
                mirrored_tf
            )
            super()._add_edge(
                from_node=mirrored_from_node,
                to_node=mirrored_to_node,
                edge=mirrored_tf,
            )
            if is_static:
                self._static_edges[mirrored_from_node][mirrored_to_node] = True

    def update_tf(self, tf: BatchFrameTransform):
        """Update a BatchFrameTransform in the graph.

        You can only update a non-static BatchFrameTransform. If the
        BatchFrameTransform is static, it will raise a ValueError.

        If the mirrored(inversed) edge exists, it will also be updated
        accordingly.

        Args:
            tf (BatchFrameTransform): The BatchFrameTransform to update.

        Raises:
            ValueError: If the transform is static, mirrored-only, missing,
                or would violate the graph batch contract.
        """
        old = self.edges.get(tf.parent_frame_id, {}).get(
            tf.child_frame_id, None
        )
        if old is None:
            is_in_mirror = self.is_mirrored_tf(
                tf.parent_frame_id, tf.child_frame_id
            )
            if is_in_mirror:
                raise ValueError(
                    f"BatchFrameTransform from {tf.parent_frame_id} to "
                    f"{tf.child_frame_id} is a mirrored transform. "
                    f"Please update the original transform from "
                    f"{tf.child_frame_id} to {tf.parent_frame_id}."
                )
            else:
                raise ValueError(
                    f"BatchFrameTransform from {tf.parent_frame_id} to "
                    f"{tf.child_frame_id} does not exist."
                )
        # check if the new transform is static
        is_static = self._static_edges.get(tf.parent_frame_id, {}).get(
            tf.child_frame_id, False
        )
        if is_static:
            raise ValueError(
                f"Cannot update static BatchFrameTransform from "
                f"{tf.parent_frame_id} to {tf.child_frame_id}."
            )
        candidate_tf_list = []
        for existing_tf in self._non_mirrored_tf_list():
            if (
                existing_tf.parent_frame_id == tf.parent_frame_id
                and existing_tf.child_frame_id == tf.child_frame_id
            ):
                candidate_tf_list.append(tf)
            else:
                candidate_tf_list.append(existing_tf)
        self._validate_tf_batch_contract(candidate_tf_list)
        # update the edge
        self.edges[tf.parent_frame_id][tf.child_frame_id] = tf
        # update the mirrored edge if it exists
        if self.is_mirrored_tf(
            tf.child_frame_id,
            tf.parent_frame_id,
        ):
            mirrored_tf = tf.inverse()
            mirrored_from_node = mirrored_tf.parent_frame_id
            mirrored_to_node = mirrored_tf.child_frame_id
            self._mirrored_edges[mirrored_from_node][mirrored_to_node] = (
                mirrored_tf
            )
            self.edges[mirrored_from_node][mirrored_to_node] = mirrored_tf

    def _add_tf(
        self,
        tf_list: Sequence[BatchFrameTransform],
        bidirectional: bool = True,
        static_tf: Sequence[bool] | None = None,
    ):
        """Add a list of BatchFrameTransform to the graph.

        This method should only be called during initialization.

        Args:
            tf_list (list[BatchFrameTransform]): A list of BatchFrameTransform
                objects to add to the graph.
            bidirectional (bool): Whether to add mirrored edges in the opposite
                direction. Defaults to True.
            static_tf (list[bool] | None): A list of booleans indicating
                whether each BatchFrameTransform is static. If None, all
                transforms are considered non-static. Defaults to None.

        Raises:
            ValueError: If the added transforms would violate the graph batch
                contract.
        """

        if static_tf is not None and len(static_tf) != len(tf_list):
            raise ValueError(
                "static_tf and tf_list must have the same length."
            )
        self._validate_tf_batch_contract(
            [*self._non_mirrored_tf_list(), *tf_list]
        )
        static_tf = static_tf or [False] * len(tf_list)
        for tf, is_static in zip(tf_list, static_tf, strict=True):
            if tf.parent_frame_id is None:
                raise ValueError(
                    "BatchFrameTransform must have a parent frame ID."
                )
            if tf.child_frame_id is None:
                raise ValueError(
                    "BatchFrameTransform must have a child frame ID."
                )
            if tf.parent_frame_id not in self.nodes:
                self._add_node(tf.parent_frame_id, tf.parent_frame_id)
            if tf.child_frame_id not in self.nodes:
                self._add_node(tf.child_frame_id, tf.child_frame_id)
            self._add_edge(
                from_node=tf.parent_frame_id,
                to_node=tf.child_frame_id,
                edge=tf,
                bidirectional=bidirectional,
                is_static=is_static,
            )

    def add_tf(
        self,
        tf: Sequence[BatchFrameTransform] | BatchFrameTransform,
        static_tf: Sequence[bool] | bool | None = None,
    ):
        """Add a list of BatchFrameTransform to the graph.

        Args:
            tf (Sequence[BatchFrameTransform]|BatchFrameTransform): The
                BatchFrameTransform objects to add to the graph.
            static_tf (Sequence[bool] | bool, None, optional): A list of
                booleans indicating whether each BatchFrameTransform is static.
                If None, all transforms are considered non-static. Defaults
                to None.

        Raises:
            ValueError: If the added transforms would violate the graph batch
                contract.
        """

        if isinstance(tf, BatchFrameTransform):
            tf = [tf]
        if isinstance(static_tf, bool):
            static_tf = [static_tf]

        return self._add_tf(
            tf_list=tf,
            static_tf=static_tf,
            bidirectional=self._bidirectional,
        )

    def get_tf(
        self, parent_frame_id: str, child_frame_id: str, compose: bool = True
    ) -> BatchFrameTransform | list[BatchFrameTransform] | None:
        """Get the transformation between two frames.

        Note:
            - If compose is True, it returns a single BatchFrameTransform
              object representing the pose of `child_frame_id` expressed in
              `parent_frame_id`. The returned BatchFrameTransform may be
              shared from graph and not copied.
            - If compose is False, it returns a list of BatchFrameTransform
              objects ordered so composing the returned list yields the same
              pose. The BatchFrameTransform objects are shared from graph and
              not copied.
            - If no path exists, it returns None.

        Args:
            parent_frame_id (str): The ID of the parent frame.
            child_frame_id (str): The ID of the child frame.

        Returns:
            BatchFrameTransform | list[BatchFrameTransform] | None: The
            transformation between the two frames. If compose is True, it
            returns a single BatchFrameTransform object. If compose is False,
            it returns a list of BatchFrameTransform objects in compose order.
            If no path exists, it returns None.
        """
        if (
            parent_frame_id not in self.nodes
            or child_frame_id not in self.nodes
        ):
            return None

        transform_chain = self.get_path_by_bfs(
            src_node_id=parent_frame_id, dst_node_id=child_frame_id
        )
        if transform_chain is None:
            return None
        assert len(transform_chain) > 0, "Path should not be empty."

        if len(transform_chain) > 1:
            # The BFS result starts at the requested parent frame and ends at
            # the requested child frame. Reverse it so the list is ordered
            # from child frame to parent frame, which is the order expected
            # by BatchFrameTransform.compose().
            transform_chain.reverse()

        if compose:
            if len(transform_chain) == 1:
                return transform_chain[0]
            else:
                return transform_chain[0].compose(*transform_chain[1:])
        else:
            return transform_chain

    def __getitem__(self, idx: list[int] | slice | int) -> Self:
        """Slice all graph edges along the effective graph batch dimension.

        Edges whose batch size already matches the effective graph batch size
        are sliced directly. Singleton edges with batch size ``1`` are kept as
        singleton edges for non-empty selections and can be materialized later
        with :meth:`repeat_singleton_tfs`. Empty slices return empty edges with
        batch size ``0``.

        Args:
            idx: Batch index selection applied to every non-mirrored edge in
                the graph. Integer indexing keeps the batch dimension,
                matching ``BatchFrameTransform.__getitem__`` semantics.

        Returns:
            Self: A new graph whose edge transforms are sliced by ``idx``.
        """
        state = self.as_state()
        graph_batch_size = self.batch_size
        sliced_batch_size = self._slice_batch_size(idx, graph_batch_size)
        sliced_tf_list = []
        for tf in state.tf_list:
            if tf.batch_size == graph_batch_size:
                sliced_tf_list.append(tf[idx])
            elif sliced_batch_size == 0:
                sliced_tf_list.append(tf[:0])
            else:
                sliced_tf_list.append(tf)
        return type(self)(
            tf_list=sliced_tf_list,
            bidirectional=state.bidirectional,
            static_tf=state.static_tf,
        )

    def as_state(self) -> BatchFrameTransformGraphState:
        edges: list[BatchFrameTransform] = []
        is_static: list[bool] = []
        for from_node, to_edges in self.edges.items():
            for to_node, edge in to_edges.items():
                if self.is_mirrored_tf(from_node, to_node):
                    continue
                edges.append(edge)
                is_static.append(self.is_static_tf(from_node, to_node))
        return BatchFrameTransformGraphState(
            tf_list=edges,
            bidirectional=self._bidirectional,
            static_tf=is_static,
        )

    @classmethod
    def from_state(
        cls,
        state: BatchFrameTransformGraphState,
    ) -> Self:
        return cls(**state.__dict__)

    def __getstate__(self) -> dict[Any, Any]:
        return self.as_state().__dict__  # type: ignore

    def __setstate__(self, state: dict[Any, Any]) -> None:
        new = BatchFrameTransformGraph.from_state(
            BatchFrameTransformGraphState(**state)
        )
        for k, v in new.__dict__.items():
            self.__dict__[k] = v

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BatchFrameTransformGraph):
            return False
        return self.as_state() == other.as_state()
