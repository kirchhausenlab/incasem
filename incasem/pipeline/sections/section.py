from abc import ABC, abstractmethod
import gunpowder as gp


class Section(ABC):
    """An interface to a sequence of ``gunpowder.`` nodes
    that is part of a ``gunpowder`` pipeline.
    This class does not handle trees of nodes.

    Nodes are stored in a dictionary, with their order in the pipeline being
    defined by the inserting order.

    """

    def __init__(self):
        self._nodes = {}
        self._new_keys = {}
        self._new_request = gp.BatchRequest()

        # TODO freeze Sections before building the pipeline like in
        # https://github.com/funkey/gunpowder/blob/master/gunpowder/freezable.py

        self._define_nodes()
        self._define_new_keys()
        self._define_new_request()

    # PUBLIC METHODS
    ################

    def insert_node_after(self, name, node, predecessor):
        """Insert an additional node into this section after predecessor.

        Args:

            name (``str``):

                Unique name of the new node.

            node (gp.BatchProvider):

                The node to be added.

            predecessor (``str``):

                Name of the node that is used as a reference for inserting.

        """
        nodes_updated = {}
        inserted = False

        for key, value in self._nodes.items():
            nodes_updated[key] = value
            if key == predecessor:
                nodes_updated[name] = node
                inserted = True

        if inserted is False:
            raise ValueError((
                f"Cannot insert {name}, predecessor {predecessor} "
                f"does not exist in {self.__class__.__name__}."
            ))

        self._nodes = nodes_updated

    def insert_node_before(self, name, node, successor):
        """Insert an additional node into this section after successor.

        Args:

            name (``str``):

                Unique name of the new node.

            node (gp.BatchProvider):

                The node to be added.

            successor (``str``):

                Name of the node that is used as a reference for inserting.

        """
        nodes_updated = {}
        inserted = False

        for key, value in self._nodes.items():
            if key == successor:
                nodes_updated[name] = node
                inserted = True
            nodes_updated[key] = value

        if inserted is False:
            raise ValueError((
                f"Cannot insert {name}, succcesor {successor} "
                f"does not exist in {self.__class__.__name__}."
            ))

        self._nodes = nodes_updated

    def add_to_pipeline(self, pipeline):
        for _, node in self._nodes.items():
            pipeline = pipeline + node
        return pipeline

    def get_pipeline(self):
        raise NotImplementedError((
            f"Class {type(self).__name__} "
            "does not implement 'get_pipeline'. Use 'add_to_pipeline' instead."
        ))

    # PROTECTED METHODS
    ###################

    @abstractmethod
    def _define_nodes(self):
        pass

    def _define_new_keys(self):
        pass

    def _define_new_request(self):
        pass

    # PROPERTIES
    ############

    @property
    def nodes(self):
        return self._nodes

    @property
    def new_keys(self):
        return self._new_keys

    @property
    def new_request(self):
        return self._new_request
