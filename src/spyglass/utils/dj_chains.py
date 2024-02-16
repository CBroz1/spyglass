from functools import cached_property
from typing import List, Union

import datajoint as dj
import networkx as nx
from datajoint.expression import QueryExpression
from datajoint.table import Table
from datajoint.utils import get_master

from spyglass.utils.dj_merge_tables import RESERVED_PRIMARY_KEY as MERGE_PK
from spyglass.utils.logging import logger


class TableChains:
    """Class for representing chains from parent to Merge table via parts.

    Functions as a plural version of TableChain, allowing a single `join`
    call across all chains from parent -> Merge table.

    Attributes
    ----------
    parent : Table
        Parent or origin of chains.
    child : Table
        Merge table or destination of chains.
    connection : datajoint.Connection, optional
        Connection to database used to create FreeTable objects. Defaults to
        parent.connection.
    part_names : List[str]
        List of full table names of child parts.
    chains : List[TableChain]
        List of TableChain objects for each part in child.
    has_link : bool
        Cached attribute to store whether parent is linked to child via any of
        child parts. False if (a) child is not in parent.descendants or (b)
        nx.NetworkXNoPath is raised by nx.shortest_path for all chains.

    Methods
    -------
    __init__(parent, child, connection=None)
        Initialize TableChains with parent and child tables.
    __repr__()
        Return full representation of chains.
        Multiline parent -> child for each chain.
    __len__()
        Return number of chains with links.
    __getitem__(index: Union[int, str])
        Return TableChain object at index, or use substring of table name.
    join(restriction: str = None)
        Return list of joins for each chain in self.chains.
    """

    def __init__(self, parent, child, connection=None):
        self.parent = parent
        self.child = child
        self.connection = connection or parent.connection
        parts = child.parts(as_objects=True)
        self.part_names = [part.full_table_name for part in parts]
        self.chains = [TableChain(parent, part) for part in parts]
        self.has_link = any([chain.has_link for chain in self.chains])

    def __repr__(self):
        return "\n".join([str(chain) for chain in self.chains])

    def __len__(self):
        return len([c for c in self.chains if c.has_link])

    def __getitem__(self, index: Union[int, str]):
        """Return FreeTable object at index."""
        if isinstance(index, str):
            for i, part in enumerate(self.part_names):
                if index in part:
                    return self.chains[i]
        return self.chains[index]

    def join(self, restriction=None) -> List[QueryExpression]:
        """Return list of joins for each chain in self.chains."""
        restriction = restriction or self.parent.restriction or True
        joins = []
        for chain in self.chains:
            if joined := chain.join(restriction):
                joins.append(joined)
        return joins


class TableChain:
    """Class for representing a chain of tables.

    A chain is a sequence of tables from parent to child identified by
    networkx.shortest_path. Parent -> Merge should use TableChains instead to
    handle multiple paths to the respective parts of the Merge table.

    Attributes
    ----------
    parent : Table
        Parent or origin of chain.
    child : Table
        Child or destination of chain.
    _connection : datajoint.Connection, optional
        Connection to database used to create FreeTable objects. Defaults to
        parent.connection.
    _link_symbol : str
        Symbol used to represent the link between parent and child. Hardcoded
        to " -> ".
    _has_link : bool
        Cached attribute to store whether parent is linked to child. False if
        child is not in parent.descendants or nx.NetworkXNoPath is raised by
        nx.shortest_path.
    names : List[str]
        List of full table names in chain. Generated by networkx.shortest_path.
    objects : List[dj.FreeTable]
        List of FreeTable objects for each table in chain.

    Methods
    -------
    __str__()
        Return string representation of chain: parent -> child.
    __repr__()
        Return full representation of chain: parent -> {links} -> child.
    __len__()
        Return number of tables in chain.
    __getitem__(index: Union[int, str])
        Return FreeTable object at index, or use substring of table name.
    join(restriction: str = None)
        Return join of tables in chain with restriction applied to parent.
    """

    def __init__(self, parent: Table, child: Table, connection=None):
        self._connection = connection or parent.connection
        if not self._connection.dependencies._loaded:
            self._connection.dependencies.load()

        if (  # if child is a merge table
            get_master(child.full_table_name) == ""
            and MERGE_PK in child.heading.names
        ):
            raise TypeError("Child is a merge table. Use TableChains instead.")

        self._link_symbol = " -> "
        self.parent = parent
        self.child = child
        self._has_link = True

    def __str__(self):
        """Return string representation of chain: parent -> child."""
        if not self._has_link:
            return "No link"
        return (
            self.parent.table_name + self._link_symbol + self.child.table_name
        )

    def __repr__(self):
        """Return full representation of chain: parent -> {links} -> child."""
        return (
            "Chain: "
            + self._link_symbol.join([t.table_name for t in self.objects])
            if self.names
            else "No link"
        )

    def __len__(self):
        """Return number of tables in chain."""
        return len(self.names)

    def __getitem__(self, index: Union[int, str]) -> dj.FreeTable:
        """Return FreeTable object at index."""
        if isinstance(index, str):
            for i, name in enumerate(self.names):
                if index in name:
                    return self.objects[i]
        return self.objects[index]

    @property
    def has_link(self) -> bool:
        """Return True if parent is linked to child.

        Cached as hidden attribute _has_link to set False if nx.NetworkXNoPath
        is raised by nx.shortest_path.
        """
        return self._has_link

    def pk_link(self, src, trg, data) -> float:
        """Return 1 if data["primary"] else float("inf").

        Currently unused. Preserved for future debugging. shortest_path accepts
        an option weight callable parameter.
        nx.shortest_path(G, source, target,weight=pk_link)
        """
        return 1 if data["primary"] else float("inf")

    def shortest_path(self, directed=True) -> List[str]:
        """Return list of full table names in chain.

        Parameters
        ----------
        directed : bool, optional
            If True, use directed graph. If False, use undirected graph.

        Uses networkx.shortest_path. Ignores numeric table names, which are
        'gaps' or alias nodes in the graph. See datajoint.Diagram._make_graph
        source code for comments on alias nodes.
        """

        graph = (
            self._connection.dependencies
            if directed
            else self._connection.dependencies.to_undirected()
        )
        try:
            return [
                name
                for name in nx.shortest_path(
                    graph,
                    self.parent.full_table_name,
                    self.child.full_table_name,
                )
                if not name.isdigit()
            ]
        except nx.NetworkXNoPath:
            return None

    @cached_property
    def names(self) -> List[str]:
        """Return list of full table names in chain.

        Tries to do a directed shortest path first, then undirected. If neither
        sets _has_link to False and returns None. Undirected permits paths to
        traverse from merge part-parent -> merge part -> merge table.
        """
        if not self._has_link:
            logger.debug(f"Early exit {self}")
            return None

        if directed_link := self.shortest_path(directed=True):
            return directed_link
        if undirected_link := self.shortest_path(directed=False):
            # Note: currently finds shortest path, which may be through
            # peripheral tables like Analysisfile.
            # Need to check that join still works for these
            logger.debug(f"Path undirected {self}")
            return undirected_link

        logger.debug(f"Set nolink {self}")
        self._has_link = False
        return None

    @cached_property
    def objects(self) -> List[dj.FreeTable]:
        """Return list of FreeTable objects for each table in chain."""
        return (
            [dj.FreeTable(self._connection, name) for name in self.names]
            if self.names
            else None
        )

    def join(
        self, restriction: str = None, reverse_order: bool = False
    ) -> dj.expression.QueryExpression:
        """Return join of tables in chain with restriction applied to parent.

        Parameters
        ----------
        restriction : str, optional
            Restriction to apply to first table in the order.
            Defaults to self.parent.restriction.
        reverse_order : bool, optional
            If True, join tables in reverse order. Defaults to False.
        """
        if not self._has_link:
            return None

        objects = self.objects[::-1] if reverse_order else self.objects
        restriction = restriction or self.parent.restriction or True
        join = objects[0] & restriction
        for table in objects[1:]:
            try:
                join = join.proj() * table
            except dj.DataJointError as e:
                attribute = str(e).split("attribute ")[-1]
                logger.error(
                    f"{str(self)} at {table.table_name} with {attribute}"
                )
                return None
        return join
