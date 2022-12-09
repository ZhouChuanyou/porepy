"""Tests of geometry part of a simulation model.

Testing covers:
    Setting of mixed-dimensional grid
    Subdomain and interface list methods:
        subdomains_to_interfaces
        interfaces_to_subdomains
    Utility methods:
        domain_boundary_sides
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


class SingleFracture2d(pp.ModelGeometry):
    """Single fracture in 2d with unstructured simplex mesh."""

    num_fracs: int = 1
    """Used to compare to the size of the fracture network assigned internally."""

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.5, "mesh_size_min": 0.5, "mesh_size_bound": 0.5}

    def set_fracture_network(self):
        pts = np.array([[0, 0.5], [0.5, 0.5]])
        edges = np.array([[0], [1]])
        domain = pp.SquareDomain([1, 1])
        self.fracture_network = pp.FractureNetwork2d(pts, edges, domain)


class TwoFractures2d(SingleFracture2d):
    """Two fractures in 2d with unstructured simplex mesh.

    One fracture is horizontal, the other is tilted (both x and y are non-zero).
    """

    num_fracs: int = 2

    def set_fracture_network(self):
        # The first fracture extends from (0, 0.5) to (0.7, 0.5).
        # The second fractures extends from (0.3, 0.3) to (0.7, 0.7).
        pts = np.array([[0, 0.7, 0.3, 0.7], [0.5, 0.5, 0.3, 0.7]])
        edges = np.array([[0, 2], [1, 3]])
        domain = pp.SquareDomain([1, 1])
        self.fracture_network = pp.FractureNetwork2d(pts, edges, domain)


class ThreeFractures3d(SingleFracture2d):
    """Three fractures in 3d with unstructured simplex mesh."""

    ambient_dimension: int = 3
    """Used to compare to the nd attribute assigned internally."""

    num_fracs: int = 3
    """Used to compare to the size of the fracture network assigned internally."""

    def set_fracture_network(self):
        coords = [0, 1]
        pts0 = [coords[0], coords[0], coords[1], coords[1]]
        pts1 = [coords[0], coords[1], coords[1], coords[0]]
        pts2 = [0.5, 0.5, 0.5, 0.5]
        fracs = [
            # A fracture with vertices (0, 0, 0.5), (0, 1, 0.5), (1, 1, 0.5), (1, 0, 0.5)
            pp.PlaneFracture(np.array([pts0, pts1, pts2])),
            # A fracture with vertexes (0.5, 0, 0), (0.5, 0, 1), (0.5, 1, 1), (0.5, 1, 0)
            pp.PlaneFracture(np.array([pts2, pts0, pts1])),
            # A fracture with vertexes (0, 0.5, 0), (1, 0.5, 0), (1, 0.5, 1), (0, 0.5, 1)
            pp.PlaneFracture(np.array([pts1, pts2, pts0])),
        ]
        domain = pp.CubeDomain([1, 1, 1])
        self.fracture_network = pp.FractureNetwork3d(fracs, domain)


class BaseWithUnits(pp.ModelGeometry):
    """ModelGeometry.set_md_geometry requires a units attribute."""

    units: pp.Units = pp.Units()


geometry_list = [BaseWithUnits, SingleFracture2d, TwoFractures2d, ThreeFractures3d]


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_set_fracture_network(geometry_class):
    geometry = geometry_class()
    geometry.set_fracture_network()
    assert getattr(geometry, "num_fracs", 0) == geometry.fracture_network.num_frac()


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_set_geometry(geometry_class):
    geometry = geometry_class()
    geometry.set_geometry()
    for attr in ["mdg", "box", "nd", "fracture_network"]:
        assert hasattr(geometry, attr)
    # For now, the default is not to assign a well network. Assert to remind ourselves to
    # add testing if default is changed.
    assert not hasattr(geometry, "well_network")

    # Checks on attribute values. Default values correspond to the un-modified
    assert geometry.nd == getattr(geometry, "ambient_dimension", 2)


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_boundary_sides(geometry_class):
    geometry = geometry_class()
    geometry.set_geometry()
    for sd in geometry.mdg.subdomains():
        all_bf, east, west, north, south, top, bottom = geometry.domain_boundary_sides(
            sd
        )
        all_bool = np.zeros(sd.num_faces, dtype=bool)
        all_bool[all_bf] = 1

        # Check that only valid boundaries are picked
        domain_or_internal_bf = np.where(np.sum(np.abs(sd.cell_faces), axis=1) == 1)
        assert np.all(np.in1d(all_bf, domain_or_internal_bf))
        frac_faces = sd.tags["fracture_faces"].nonzero()[0]
        assert np.all(np.logical_not(np.in1d(all_bf, frac_faces)))
        assert np.all(all_bool == (east + west + north + south + top + bottom))

        # Check coordinates
        for side, dim in zip([east, north, top], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], 1))
        for side, dim in zip([west, south, bottom], [0, 1, 2]):
            assert np.all(np.isclose(sd.face_centers[dim, side], 0))


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_wrap_grid_attributes(geometry_class: type[pp.ModelGeometry]) -> None:
    """Test that the grid attributes are wrapped correctly.

    The test is based on sending in a list of grids (both subdomains and interfaces)
    wrap a number of attributes, and check that the attributes are wrapped correctly.

    """
    geometry = geometry_class()
    geometry.set_geometry()
    nd: int = geometry.nd

    # Various combinations of single and many subdomains
    all_subdomains = geometry.mdg.subdomains()
    top_subdomain = geometry.mdg.subdomains(dim=geometry.nd)
    some_subdomains = top_subdomain + geometry.mdg.subdomains(dim=geometry.nd - 1)
    # An empty list
    empty_subdomains: list[pp.Grid] = []

    # Various combinations of single and many interfaces
    all_interfaces = geometry.mdg.interfaces()
    top_interfaces = geometry.mdg.interfaces(dim=geometry.nd - 1)
    some_interfaces = top_interfaces + geometry.mdg.interfaces(dim=geometry.nd - 2)

    # Gather all lists of subdomains and all lists of interfaces
    test_subdomains = [all_subdomains, top_subdomain, some_subdomains, empty_subdomains]
    test_interfaces = [all_interfaces, top_interfaces, some_interfaces]

    # Equation system, needed for evaluation.
    eq_system = pp.ad.EquationSystem(geometry.mdg)

    # Test that an error is raised if the grid does not have such an attribute
    with pytest.raises(ValueError):
        geometry.wrap_grid_attribute(top_subdomain, "no_such_attribute")
    # Test that the an error is raised if we try to wrap a field which is not an
    # ndarray.
    with pytest.raises(ValueError):
        # This will return a string
        geometry.wrap_grid_attribute(top_subdomain, "name")

    # One loop for both subdomains and interfaces.
    for grids in test_subdomains + test_interfaces:

        # Which attributes to test depends on whether the grids are subdomains or
        # interfaces.
        if len(grids) == 0 or isinstance(grids[0], pp.MortarGrid):
            # Also include the empty list here, one attribute should be sufficient to
            # test that a zero matrix is returned.
            attr_list = ["cell_centers"]
            dim_list = [nd]
        else:
            # All relevant attributes for subdomain grids
            attr_list = [
                "cell_centers",
                "face_centers",
                "face_normals",
                "cell_volumes",
                "face_areas",
            ]
            #  List of dimensions, corresponding to the order in attr_list
            dim_list = [nd, nd, nd, 1, 1]

        # Loop over attributes and corresponding dimensions.
        for attr, dim in zip(attr_list, dim_list):
            # Get hold of the wrapped attribute and the wrapping with inverse=True
            wrapped_value = geometry.wrap_grid_attribute(grids, attr, dim=dim).evaluate(
                eq_system
            )
            wrapped_value_inverse = geometry.wrap_grid_attribute(
                grids, attr, dim=dim, inverse=True
            ).evaluate(eq_system)

            # Check that the wrapped attribute is a matrix
            assert isinstance(wrapped_value, sps.spmatrix)
            assert isinstance(wrapped_value_inverse, sps.spmatrix)

            # Check that the matrix have the expected size, which depends on the type
            # of attribute wrapped (cell or face) and the dimension of the field.
            size_key = "num_cells" if "cell" in attr else "num_faces"
            tot_size = sum([getattr(sd, size_key) for sd in grids])

            assert wrapped_value.shape == (tot_size * dim, tot_size * dim)
            assert wrapped_value_inverse.shape == (tot_size * dim, tot_size * dim)

            # Get hold of the actual attribute values; we know these reside on the
            # main diagonal.
            values = wrapped_value.diagonal()
            values_inverse = wrapped_value_inverse.diagonal()

            # Counter for the current position in the wrapped attribute
            ind_cc = 0

            # Loop over the grids (be they subdomains or interfaces)
            for grid in grids:
                # Get hold of the actual attribute values straight from the grid
                size = getattr(grid, size_key)
                # Note the use of 2d here, or else the below accessing of [:dim] would
                # not work.
                actual_value = np.atleast_2d(getattr(grid, attr))
                # Compare values with the wrapped attribute, both usual and inverse.
                assert np.allclose(
                    values[ind_cc : ind_cc + size * dim],
                    actual_value[:dim].ravel("F"),
                )
                assert np.allclose(
                    values_inverse[ind_cc : ind_cc + size * dim],
                    1 / actual_value[:dim].ravel("F"),
                )
                # Move to the new position in the wrapped attribute
                ind_cc += size * dim


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_subdomain_interface_methods(geometry_class: type[pp.ModelGeometry]) -> None:
    """Test interfaces_to_subdomains and subdomains_to_interfaces.

    Parameters:
        geometry_class:

    """
    geometry = geometry_class()
    geometry.set_geometry()
    all_subdomains = geometry.mdg.subdomains()
    all_interfaces = geometry.mdg.interfaces()

    returned_subdomains = geometry.interfaces_to_subdomains(all_interfaces)
    returned_interfaces = geometry.subdomains_to_interfaces(all_subdomains)
    if all_interfaces == []:
        assert returned_subdomains == []
        assert returned_interfaces == []
    else:
        assert all_subdomains == returned_subdomains
        assert all_interfaces == returned_interfaces

    # Empty list passed should return empty list for both methods.
    no_subdomains = geometry.interfaces_to_subdomains([])
    no_interfaces = geometry.subdomains_to_interfaces([])
    assert no_subdomains == []
    assert no_interfaces == []
    if getattr(geometry, "num_fracs", 0) > 1:
        # Matrix and two fractures. TODO: Use three_sds?
        three_sds = all_subdomains[:2]

        two_fractures = all_subdomains[1:3]
        # Only those interfaces involving one of the two fractures are expected.
        interfaces = []
        for sd in two_fractures:
            interfaces += geometry.mdg.subdomain_to_interfaces(sd)
        sorted_interfaces = geometry.mdg.sort_interfaces(interfaces)
        assert sorted_interfaces == geometry.subdomains_to_interfaces(two_fractures)


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_internal_boundary_normal_to_outwards(geometry_class: type[pp.ModelGeometry]):
    # Define the geometry
    geometry: pp.ModelGeometry = geometry_class()
    geometry.set_geometry()
    dim = geometry.nd

    # Make an equation system, which is needed for parsing of the Ad operator
    # representations of the geometry
    eq_sys = pp.EquationSystem(geometry.mdg)

    # The function to be tested only accepts the top level subdomain(s)
    # NOTE: This test does not cover the case of multiple subdomains on the top level,
    # as could happen if we implement a domain decomposition approach. We could make
    # a partitioning of the top dimensional grid and thereby test the functionality,
    # but this has not been prioritized. Passing in the same subdomain twice will not
    # work, since the function will uniquify the input.
    subdomains = [
        geometry.mdg.interface_to_subdomain_pair(intf)[0]
        for intf in geometry.mdg.interfaces()
    ]

    # Get hold of the matrix to be tested, parse it to numerical format.
    sign_switcher = geometry.internal_boundary_normal_to_outwards(subdomains, dim=dim)
    mat = sign_switcher.evaluate(eq_sys)

    # Check that the wrapped attribute is a matrix
    assert isinstance(mat, sps.spmatrix)
    # Check that the matrix have the expected size.
    expected_size = sum([sd.num_faces for sd in subdomains]) * dim
    assert mat.shape == (expected_size, expected_size)

    # All values are stored on the main diagonal, fetch these.
    mat_vals = mat.diagonal()

    # Offset, needed to deal with the case of several subdomains. It is not relevant
    # for now (see comment above), but we keep it.
    offset = 0

    # Loop over subdomains (at the time of writing, there will only be one) and check
    # that the values are as expected.
    for sd in subdomains:
        # We get the expected values from the cell-face relation of the subdomain:
        # By assumptions in the mesh construction, the normal vector of a boundary
        # face is pointing outwards for those faces that have a positive cell-face
        # item (note that on boundary faces, there is only one non-zero entry in the
        # cell-face for each row, ie., each face).
        cf = sd.cell_faces
        # Summing is a trick to get the sign of the cell-face relation for the boundary
        # faces (we don't care about internal faces).
        cf_sum = np.sum(cf, axis=1)
        # Only compare for fracture faces
        fracture_faces = np.where(sd.tags["fracture_faces"])[0]
        # The matrix constrained to this subdomain
        loc_vals = mat_vals[offset : offset + sd.num_faces * dim]
        loc_size = loc_vals.size
        # The matrix will have one row for each face for each dimension. Loop over the
        # dimensions; the sign should be the same for all dimensions.
        for i in range(dim):
            # Indices belonging to the current dimension
            dim_ind = np.arange(i, loc_size, dim)
            dim_vals = loc_vals[dim_ind]
            assert np.allclose(
                dim_vals[fracture_faces], cf_sum[fracture_faces].A.ravel()
            )
        # Update offset, needed to test for multiple subdomains.
        offset += sd.num_faces * dim


# test_internal_boundary_normal_to_outwards(geometry_list[2])


@pytest.mark.parametrize("geometry_class", geometry_list)
def test_outwards_normals(geometry_class: type[pp.ModelGeometry]) -> None:
    """Test :meth:`pp.ModelGeometry.outwards_internal_boundary_normals`.

    Parameters:
        geometry_class: Class to test.

    """
    # Define the geometry
    geometry: pp.ModelGeometry = geometry_class()
    geometry.set_geometry()
    dim = geometry.nd
    # Make an equation system, which is needed for parsing of the Ad operator
    # representations of the geometry
    eq_sys = pp.EquationSystem(geometry.mdg)

    # First check the method to compute
    interfaces = geometry.mdg.interfaces()
    normal_op = geometry.outwards_internal_boundary_normals(interfaces, unitary=True)
    normals = normal_op.evaluate(eq_sys)

    # The result should be a sparse matrix
    assert isinstance(normals, sps.spmatrix)

    if len(interfaces) == 0:
        # We have checked that the method can handle empty lists (parsable operator).
        # Check the entry and exit.
        assert np.allclose(normals.A, 0)
        return

    diag = normals.diagonal()
    # Check that all off-diagonal entries are zero
    assert np.allclose(np.diag(diag) - normals, 0)

    # Convert the normals into a nd x num_faces array
    normals_reshaped = np.reshape(diag, (dim, -1), order="F")

    # Check that the normals are unit vectors
    assert np.allclose(np.linalg.norm(normals_reshaped, axis=0), 1)

    # Check that the normals are outward. This is done by checking that the dot product
    # of the normal and the vector from the center of the interface to the center of the
    # neighboring subdomain cell is positive.
    offset = 0
    for intf in interfaces:
        sd = geometry.mdg.interface_to_subdomain_pair(intf)[0]

        loc_normals = normals_reshaped[:, offset : offset + intf.num_cells]

        fracture_faces = intf.mortar_to_primary_avg().tocsc().indices
        proj_normals = (intf.mortar_to_primary_avg() * loc_normals.T)[fracture_faces]

        cc = sd.cell_centers
        fc = sd.face_centers

        fracture_cells = sd.cell_faces[fracture_faces].tocsr().indices

        vec = cc[:, fracture_cells] - fc[:, fracture_faces]

        nrm1 = np.linalg.norm(proj_normals, axis=1)

        nrm2 = np.linalg.norm(proj_normals + 1e-3 * vec[:dim].T, axis=1)

        assert np.all(nrm1 > nrm2)
        offset += intf.num_cells

    # Left multiply with dim-vector defined on the interface. This should give a vector
    # of length dim*num_intf_cells.
    size = dim * sum([intf.num_cells for intf in interfaces])
    dim_vec = pp.ad.Array(np.ones(size))
    product = (normal_op * dim_vec).evaluate(eq_sys)
    assert product.shape == (size,)
    inner_product = np.sum(product.reshape((dim, -1), order="F"), axis=0)
    # assert np.allclose(np.abs(inner_product), 1)
    # The following operation is used in models, and is therefore tested here.
    # TODO: Extract method for inner product using a basis?
    basis = geometry.basis(interfaces, dim)
    nd_to_scalar_sum = sum([e.T for e in basis])
    inner_op = nd_to_scalar_sum * (normal_op * dim_vec)
    assert np.allclose(inner_op.evaluate(eq_sys), inner_product)


test_outwards_normals(geometry_list[3])
