
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
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.01, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    SinglePhaseFlow):
    """Combining the modified geometry and the default model."""
    ...



# Pressure fuction at boundary
def p_D(xv):
    x,y,z = xv
    val = (1 - x)*x*(y**2)*np.sin(2*np.pi*x)
    return val

def f_rhs(xv):
    x,y,z = xv
    val = 4*np.pi*(-1 + 2*x)*(y**2)*np.cos(2*np.pi*x) + 2*((y**2) + (-1 + x)*x*(1 - 2*(np.pi**2)*(y**2)))*np.sin(2*np.pi*x)
    return val

def qn_at_top_bc(xv):
    x,y,z = xv
    val = -2*(1 - x)*x*y*np.sin(2*np.pi*x)
    return val

class ModifiedBC(BoundaryConditionsSinglePhaseFlow):

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.all_bf, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        xc = boundary_grid.cell_centers.T
        values[bounds.all_bf] = np.array(list(map(p_D, xc)))
        return values

    # def bc_values_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
    #     """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
    #     bounds = self.domain_boundary_sides(sd)
    #     bc = pp.BoundaryCondition(sd, bounds.west, "dir")
    #     return bc


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

# pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), linewidth=0.25, title="Pressure distribution", plot_2d=True)
