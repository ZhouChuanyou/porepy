
import porepy as pp
import numpy as np
import scipy.sparse as sps
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
        cell_size = self.solid.convert_units(0.001, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class SinglePhaseFlowGeometry(
    ModifiedGeometry,
    SinglePhaseFlow):
    """Combining the modified geometry and the default model."""
    ...

# Paremeter to control the non-linearity in rho, eta in [0,2000]
global_eta = 100.0

# Manufacture solution data
def p_exact(xv):
    x,y,z = xv
    val = (1 - x)*x*(y**2)*np.sin(2*np.pi*x)
    return val

# Right hand side source
def f_rhs(xv, m_eta = global_eta):
    x,y,z = xv
    val = 2*((-1 + x)*x*np.sin(2*np.pi*x) + 5*((-1 + x)**3)*(x**3)*(y**4)*m_eta*
      (np.sin(2*np.pi*x)**3) + (-1 + x)*x*(y**6)*m_eta*np.sin(2*np.pi*x)*
      ((2*np.pi*(-1 + x)*x*np.cos(2*np.pi*x) + (-1 + 2*x)*np.sin(2*np.pi*x))**2) +
     (y**2)*(2*np.pi*(1 - 2*x)*np.cos(2*np.pi*x) +
        (-1 + 2*(np.pi**2)*(-1 + x)*x)*np.sin(2*np.pi*x))*
      (-1 - ((-1 + x)**2)*(x**2)*(y**4)*m_eta*(np.sin(2*np.pi*x)**2)))
    return val

# Normal flux at y = 1.0
def qn_at_top_bc(xv, m_eta = global_eta):
    x,y,z = xv
    val = 2*(1 - x)*x*y*np.sin(2*np.pi*x)*(-1 - ((1 - x)**2)*(x**2)*(y**4)*m_eta*(np.sin(2*np.pi*x)**2))
    return val

class ModifiedBC(BoundaryConditionsSinglePhaseFlow):

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west, south and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc_idx = bounds.west + bounds.south + bounds.east + bounds.north
        bc = pp.BoundaryCondition(sd, bc_idx, "dir")
        return bc

    # def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
    #     """Assign dirichlet to the west, south and east boundaries. The rest are Neumann by default."""
    #     bounds = self.domain_boundary_sides(sd)
    #     bc_idx = bounds.west + bounds.south + bounds.east + bounds.north
    #     bc = pp.BoundaryCondition(sd, bc_idx, "dir")
    #     return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        bounds = self.domain_boundary_sides(boundary_grid)
        bc_idx = bounds.west + bounds.south + bounds.east + bounds.north
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
    #
    # def bc_values_fluid_flux(self, boundary_grid: pp.BoundaryGrid) -> pp.BoundaryCondition:
    #     bounds = self.domain_boundary_sides(boundary_grid)
    #     bc_idx = bounds.north
    #     values = np.zeros(boundary_grid.num_cells)
    #     xc = boundary_grid.cell_centers.T
    #     values[bc_idx] = (boundary_grid.cell_volumes * np.array(list(map(qn_at_top_bc, xc))))[bc_idx]
    #     return values


class ModifiedSource:
    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assign unitary fracture source"""
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

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
    pp.constitutive_laws.DarcysLawAd,
    ModifiedGeometry,
    ModifiedSource,
    ModifiedBC,
    SinglePhaseFlow):
    """Adding both geometry and modified boundary conditions to the default model."""

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd | pp.ad.TpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        if self.params["base_discr"] == "tpfa":
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def initial_condition(self):
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            p_e = np.array(list(map(p_exact, sd.cell_centers.T)))
            pp.set_solution_values(
                name=self.pressure_variable,
                values=p_e,
                data=data,
                iterate_index=0,
                time_step_index=0,
            )

    def fluid_density(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rho_ref = pp.ad.Scalar(1.0, "reference_fluid_density")
        eta = pp.ad.Scalar(global_eta, "nonlinear_scale")
        # dp = self.perturbation_from_reference("pressure", subdomains)
        p = self.pressure(subdomains)
        rho = rho_ref + eta * p**2
        rho.set_name("fluid_density")
        return rho

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor. Depends on pressure.

        NOTE: *Do not* change this code without also updating the permeability in the
        test function.
        """
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)

        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        tensor_dim = 3**2

        # Set constant component of the permeability
        all_vals = np.zeros(nc * tensor_dim, dtype=float)
        all_vals[0::tensor_dim] = 1
        all_vals[4::tensor_dim] = 1
        Kappa = pp.wrap_as_dense_ad_array(all_vals, name="Spatial_permeability_component")

        e_xx = self.e_i(subdomains, i=0, dim=tensor_dim)
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)

        nonlinear_diffusion = (
                (e_xx + e_yy) @ self.fluid_density(subdomains)
        )
        nonlinear_diffusion.set_name("State_dependent_mobility_component")
        nonlinear_diffusion_weight = Kappa * ((e_xx + e_yy) @ self.fluid_density(subdomains))
        return nonlinear_diffusion_weight

    def after_nonlinear_iteration(self, solution_vector: np.ndarray):
        self._nonlinear_iteration += 1
        self.equation_system.shift_iterate_values()
        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, iterate_index=0
        )
        self.set_discretization_parameters()
        self.discretize()

params = {}
fluid_constants = pp.FluidConstants({"viscosity": 1.0, "density": 1.0})
solid_constants = pp.SolidConstants({"permeability": 1.0, "porosity": 0.0})
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
# params = {"material_constants": material_constants, "max_iterations": 50, "prepare_simulation": False, "nl_convergence_tol": 1.0e0}
params = {"base_discr": "mpfa","material_constants": material_constants, "max_iterations": 20, "prepare_simulation": False}

model = SinglePhaseFlowGeometryBC(params)

# project exact solution and used as initial guess
dimension = 2
model.prepare_simulation()
# data = model.mdg.subdomains(True, 2)
# p_e = np.array(list(map(p_exact, data[0][0].cell_centers.T)))
# data[0][1]['time_step_solutions']['pressure'][0] = p_e
# data[0][1]['iterate_solutions']['pressure'][0] = p_e

# perform assertion
jac, res = model.equation_system.assemble(evaluate_jacobian=True)
print("Residual norm at projected solution: ", np.linalg.norm(res))
almost_satisfy_pde_at_proj_solution_q = np.linalg.norm(res) < 1.0e-1


pp.run_time_dependent_model(model, params)
model.exporter.write_vtu([model.pressure_variable])
jac, res = model.equation_system.assemble(evaluate_jacobian=True)
print("Residual norm: ", np.linalg.norm(res))
consistent_discretizaiton_q = almost_satisfy_pde_at_proj_solution_q and np.linalg.norm(res) < 1.0e-13
print("Consistent discretization?", consistent_discretizaiton_q)
print("Solving n_dof:", model.equation_system.num_dofs())

# Some remarks:
# For this particular setting:
# - The problem being approximated is nonlinear in the scalar field (pressure);
# - The Mass flux and Volumetric Darcy flux coincide;
# - By employing the method bc_values_darcy_flux leads to the correct approximation;
#   while employing the method bc_values_fluid_flux leads to an incorrect approximation.
# - The parameter eta scales the nonlinear term in the density. Now for bc_values_darcy_flux
#   The approximation is incorrect due to some artificial numerical diffusion.

# Computing relative l2 error
data = model.mdg.subdomains(True, dimension)
p_h = data[0][1]['time_step_solutions']['pressure'][0]
p_e = np.array(list(map(p_exact, data[0][0].cell_centers.T)))
sol_norm = np.sqrt(np.sum((p_e) * (p_e) * data[0][0].cell_volumes))
rel_error = (p_h - p_e) / sol_norm
rel_l2_error = np.sqrt(np.sum(rel_error * rel_error * data[0][0].cell_volumes))
print("Eta value : ", global_eta)
print("Exact solution norm : ", sol_norm)
print("Relative l2_error in pressure: ", rel_l2_error)

assert consistent_discretizaiton_q



