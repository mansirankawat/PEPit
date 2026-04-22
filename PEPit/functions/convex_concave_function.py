import numpy as np

from PEPit.function import Function
from PEPit.block_partition import BlockPartition


class ConvexConcaveFunction(Function):
    """
    The :class:`ConvexConcaveFunction` class overwrites the `add_class_constraints` method of :class:`Function`,
    by implementing necessary constraints for interpolation of the class of convex-concave functions.
    Attributes:
        partition (BlockPartition): partitioning of the two sets of variables (in blocks).

    General Convex-Concave functions are not characterized by any parameter but we need to pass a two block
    partition to deal with x and y variables separately, hence can be instantiated as

    Example:
        >>> from PEPit import PEP
        >>> from PEPit.functions import ConvexConcaveFunction
        >>> problem = PEP()
        >>> func = problem.declare_function(function_class=ConvexConcaveFunction)

    References:

    `[1] Krivchenko, V. O., Gasnikov, A. V., & Kovalev, D. A. (2024). 
     Convex-concave interpolation and application of PEP to the bilinear-coupled saddle point problem.
     Russian Journal of Nonlinear Dynamics, 20(5), 875-893.`_

    """

    def __init__(self,
                 partition,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=True,
                 name=None):
        """
        Args:
            partition (BlockPartition): a :class:`BlockPartition`. 
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): Decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        Note:
            Convex-concave functions are necessarily differentiable, hence `reuse_gradient` is set to True.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

        # Create a block partition to handle convex-concave function
        assert isinstance(partition, BlockPartition)
        assert partition.get_nb_blocks() == 2
        self.partition = partition


    def add_class_constraints(self):
        """
        Formulates the list of necessary constraints for interpolation for self (convex-concave function);
        see [1, Theorem 1].

        """
        # Set function ID
        function_id = self.get_name()
        if function_id is None:
            function_id = "Function_{}".format(self.counter)

        # Set tables_of_constraints attributes
        self.tables_of_constraints["convexity_concavity_{}".format(0)] = [[]]*len(self.list_of_points)

        # Browse list of points and create interpolation constraints
        for i, point_i in enumerate(self.list_of_points):

            zi, gi, fi = point_i
            zi_id = zi.get_name()
            if zi_id is None:
                zi_id = "Point_{}".format(i)

            for j, point_j in enumerate(self.list_of_points):

                zj, gj, fj = point_j
                zj_id = zj.get_name()
                if zj_id is None:
                    zj_id = "Point_{}".format(j)

                if point_i == point_j:
                    self.tables_of_constraints["convexity_concavity_{}".format(0)][i].append(0)

                else:
                    gj_x = self.partition.get_block(gj, 0)
                    xi = self.partition.get_block(zi, 0)
                    xj = self.partition.get_block(zj, 0)

                    gi_y = self.partition.get_block(gi, 1)
                    yi = self.partition.get_block(zi, 1)
                    yj = self.partition.get_block(zj, 1)

                    constraint = fi >= fj + gj_x * (xi - xj) + gi_y * (yi - yj)

                    constraint.set_name("IC_{}_convexity_concavity_{}({}, {})".format(function_id, 0,
                                                                                  zi_id, zj_id))
                    self.tables_of_constraints["convexity_concavity_{}".format(0)][i].append(constraint)
                    self.list_of_class_constraints.append(constraint)
                    