
import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from porepy.applications.md_grids.domains import nd_cube_domain

class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.solid.convert_units(1, "m")
        dimension = 2
        self._domain = nd_cube_domain(dimension, size)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.01, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    SinglePhaseFlow):
    """Combining the modified geometry and the default model."""
    ...

# Manufacture solution data
def p_exact(xv):
    x,y,z = xv
    val = (1 - x)*x*(y**2)*np.sin(2*np.pi*x)
    return val

# right hand side source
def f_rhs(xv):
    x,y,z = xv
    val = 4*np.pi*(-1 + 2*x)*(y**2)*np.cos(2*np.pi*x) + 2*((y**2) + (-1 + x)*x*(1 - 2*(np.pi**2)*(y**2)))*np.sin(2*np.pi*x)
    return val

# normal flux at y = 1.0
def qn_at_top_bc(xv):
    x,y,z = xv
    val = -2*(1 - x)*x*y*np.sin(2*np.pi*x)
    return val

class ModifiedBC(BoundaryConditionsSinglePhaseFlow):

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west, south and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc_idx = bounds.west + bounds.south + bounds.east
        bc = pp.BoundaryCondition(sd, bc_idx, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        bc_idx = bounds.west + bounds.south + bounds.east
        values = np.zeros(boundary_grid.num_cells)
        xc = boundary_grid.cell_centers.T
        values[bc_idx] = np.array(list(map(p_exact, xc)))[bc_idx]
        return values

    # def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> pp.BoundaryCondition:
    #     bounds = self.domain_boundary_sides(boundary_grid)
    #     bc_idx = bounds.north
    #     values = np.zeros(boundary_grid.num_cells)
    #     xc = boundary_grid.cell_centers.T
    #     values[bc_idx] = (boundary_grid.cell_volumes * np.array(list(map(qn_at_top_bc, xc))))[bc_idx]
    #     return values

    def bc_values_fluid_flux(self, boundary_grid: pp.BoundaryGrid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(boundary_grid)
        bc_idx = bounds.north
        values = np.zeros(boundary_grid.num_cells)
        xc = boundary_grid.cell_centers.T
        values[bc_idx] = (boundary_grid.cell_volumes * np.array(list(map(qn_at_top_bc, xc))))[bc_idx]
        return values


class ModifiedSource:
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assign unitary fracture source"""
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []

        for sd in subdomains:
            if sd.dim == self.mdg.dim_max():
                xc = sd.cell_centers.T
                values = sd.cell_volumes * np.array(list(map(f_rhs, xc)))
            else:
                values.append(np.zeros(sd.num_cells))

        external_sources = pp.wrap_as_dense_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source

class SinglePhaseFlowGeometryBC(
    ModifiedGeometry,
    ModifiedSource,
    ModifiedBC,
    SinglePhaseFlow):
    """Adding both geometry and modified boundary conditions to the default model."""
    ...


params = {}
fluid_constants = pp.FluidConstants({"viscosity": 1.0, "density": 1.0})
solid_constants = pp.SolidConstants({"permeability": 1.0, "porosity": 0.0})
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
params = {"material_constants": material_constants}

model = SinglePhaseFlowGeometryBC(params)
pp.run_time_dependent_model(model, params)
model.exporter.write_vtu([model.pressure_variable])

# Notes about what is being inconsistent:
# For this particular setting:
# - The problem being approximated is linear the scalar field (pressure)
# - The Mass flux and Volumetric Darcy flux coincides.
# - Only employing the method bc_values_darcy_flux leads to the correct approximation,
#   while employing the method bc_values_fluid_flux leads to an incorrect approximation.

# Computing relative l2 error
dimension = 2
data = model.mdg.subdomains(True, dimension)
p_h = data[0][1]['time_step_solutions']['pressure'][0]
p_e = np.array(list(map(p_exact, data[0][0].cell_centers.T)))
sol_norm = np.sqrt(np.sum((p_e) * (p_e) * data[0][0].cell_volumes))
rel_error = (p_h - p_e) / sol_norm
rel_l2_error = np.sqrt(np.sum(rel_error * rel_error * data[0][0].cell_volumes))
print("Exact solution norm : ", sol_norm)
print("Relative l2_error in pressure: ", rel_l2_error)