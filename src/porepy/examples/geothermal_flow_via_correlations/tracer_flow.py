"""Example implementing a multi-phase multi component flow of H2O-NaCl using Driesner
correlations and a tracer-like as constitutive descriptions.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturation, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg
import scipy.sparse as sps
from scipy.sparse.csgraph import reverse_cuthill_mckee
from LinearTracerModelConfiguration import LinearTracerFlowModel as FlowModel

import porepy as pp

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def matrix_plot(matrix):
    # Define a custom colormap for divergence
    cmap_colors = [(0.0, "#4575b4"), (0.5, "white"), (1.0, "#d73027")]
    cmap_name = 'divergence_colormap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors)

    # Plot the matrix
    plt.imshow(matrix, cmap=custom_cmap, interpolation='nearest')

    # Highlight zero values in white
    plt.imshow(matrix == 0, cmap='gray', interpolation='nearest', alpha=0.3)
    # Add colorbar
    plt.colorbar()
    plt.show()

day = 86400
t_scale = 0.01
time_manager = pp.TimeManager(
    schedule=[0.0, 10.0 * day * t_scale],
    dt_init=1.0 * day * t_scale,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {"permeability": 9.869233e-14, "porosity": 0.2, "thermal_conductivity": 1.92}
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "petsc_solver_q": True,
    "nl_convergence_tol": 1.0e-3,
    "nl_convergence_tol_res": 1.0e-3,
    "max_iterations": 25,
}


class GeothermalFlowModel(FlowModel):

    def after_nonlinear_convergence(self) -> None:
        tb = time.time()
        _, res = model.equation_system.assemble(evaluate_jacobian=True)
        res_norm = np.linalg.norm(res)
        te = time.time()
        print("Elapsed time residual assemble: ", te - tb)
        print("Time step converged with residual norm: ", res_norm)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")
        super().after_nonlinear_convergence()
        self.progress_secondary_quantities_in_time()

    def after_simulation(self):
        self.exporter.write_pvd()

    def build_permutations(self):
        # Build permutations
        self.row_perm = np.concatenate(
            list(self.equation_system.assembled_equation_indices.values()))

        p_names = self.primary_variable_names + [self.secondary_variables_names[0]]
        s_names = self.secondary_variables_names
        p_dof = self.equation_system.dofs_of(p_names)
        s_names = [s_names[1], s_names[2]]
        s_dof = self.equation_system.dofs_of(s_names)
        self.col_perm = np.concatenate([p_dof, s_dof])

        fields_idx = np.add.accumulate(
            [0] + [len(p_dof), len(s_dof)])
        self.general_p_idx = np.array(range(fields_idx[0], fields_idx[1]),
                                 dtype=np.int32)
        self.general_s_idx = np.array(range(fields_idx[1], fields_idx[2]),
                                 dtype=np.int32)



    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        petsc_solver_q = self.params.get("petsc_solver_q", False)
        tb = time.time()
        if petsc_solver_q:
            self.build_permutations()
            from petsc4py import PETSc
            direct_solver_q = True

            csr_mat, res_g = self.linear_system

            # Apply permutations
            csr_mat = csr_mat[self.row_perm,:][:, self.col_perm]

            # Solving ls
            st = time.time()

        if direct_solver_q:

            Kss = csr_mat[self.general_s_idx, :][:, self.general_s_idx]
            Ksp = csr_mat[self.general_s_idx, :][:, self.general_p_idx]
            Kpp = csr_mat[self.general_p_idx, :][:, self.general_p_idx]
            Kps = csr_mat[self.general_p_idx, :][:, self.general_s_idx]

            res_p = res_g[self.general_p_idx]
            res_s = res_g[self.general_s_idx]

            Kss_inv = Kss
            jac_red = Kpp - Kps @ Kss_inv @ Ksp
            r_red = res_p - Kps @ Kss_inv @ res_s

            jac_g = PETSc.Mat().createAIJ(
                size=csr_mat.shape,
                csr=((jac_red.indptr, jac_red.indices, jac_red.data)),
            )

            ksp = PETSc.KSP().create()
            ksp.setOperators(jac_g)
            b = jac_g.createVecLeft()
            b.array[:] = r_red
            x = jac_g.createVecRight()

            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")

            ksp.setFromOptions()
            ksp.solve(b, x)

            inv_perm = pp.matrix_operations.invert_permutation(self.col_perm)
            sol = x.array[inv_perm]

        else:
            sol = super().solve_linear_system()

        reduce_linear_system_q = self.params.get("reduce_linear_system_q", False)
        if reduce_linear_system_q:
            raise ValueError("Case not implemented yet.")
        te = time.time()
        print("Elapsed time linear solve: ", te - tb)
        return sol




model = GeothermalFlowModel(params)

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)
