"""
This module contains tests that are devised to guarantee a correct integration between of the
TimeManager class into the PorePy models.

Brief summary of tests:
    * test_default_behaviour: Tests default behaviour of the TimeManager in all models. This
        also includes static models.
    * test_constant_dt: Tests time management when constant time step is used on all
        time-dependent models.
    * test_linear_non_constant_dt: Tests time management on time-dependent linear models when
        non-constant time step is used.
    * test_time_step_adaptation_after_convergence: Tests time adaptation on non-linear
        time-dependent models where convergence is achieved in all time steps.
    * test_time_step_adapation_after_failure: Tests time adaptation on non-linear
        time-dependent models where recomputation-after-failed-to-converge is necessary in at
        least one time level.

"""

import numpy as np
import pytest

import porepy as pp
from tests.common.slightly_compressible_flow_examples import NonLinearSCF

# Model classes to be tested
all_models = [
    pp.IncompressibleFlow,
    pp.SlightlyCompressibleFlow,
    pp.ContactMechanics,
    pp.ContactMechanicsBiot,
    pp.THM,
]

time_dependent_models = [pp.SlightlyCompressibleFlow, pp.ContactMechanicsBiot, pp.THM]

non_linear_models = [NonLinearSCF]


# -----> Testing default behaviour
@pytest.mark.parametrize("model_class", all_models)
def test_default_behaviour(model_class):
    """Test the default behaviour of the models"""

    model = model_class(params={"use_ad": True})

    if not model._is_time_dependent():
        pp.run_stationary_model(model, model.params)
        assert not hasattr(model, "time_manager")
    else:
        pp.run_time_dependent_model(model, model.params)
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 1
        assert model.time_manager.time == 1
        assert model.time_manager.dt == 1
        assert model._ad.time_step.evaluate(model.dof_manager) == model.time_manager.dt


# -----> Testing constant time step
tm0 = pp.TimeManager([0, 1, 2], 1, constant_dt=True)
tm1 = pp.TimeManager([0, 0.01, 0.02, 0.05], 0.01, constant_dt=True)
tm2 = pp.TimeManager([0, pp.HOUR, 2 * pp.HOUR, 10 * pp.HOUR], pp.HOUR, constant_dt=True)


@pytest.mark.parametrize("model_class", time_dependent_models)
@pytest.mark.parametrize("time_manager", [tm0, tm1, tm2])
def test_constant_dt(request, model_class, time_manager):
    """Test models when constant time step is used"""

    model = model_class(params={"use_ad": True, "time_manager": time_manager})
    pp.run_time_dependent_model(model, model.params)
    if "0" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 2
        assert model.time_manager.dt == 1
        assert model.time_manager.time == 2
    elif "1" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 0.05
        assert model.time_manager.dt == 0.01
        assert model.time_manager.time == 0.05
    elif "2" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 10 * pp.HOUR
        assert model.time_manager.dt == pp.HOUR
        assert model.time_manager.time == 10 * pp.HOUR


# -----> Testing linear models
# TODO: Replace [pp.SlightlyCompressibleFlow] by time_dependent_models when
@pytest.mark.parametrize("model_class", [pp.SlightlyCompressibleFlow])
def test_linear_non_constant_dt(model_class):
    """Test linear models when non-constant time steps are used"""
    time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.01)
    model = model_class(params={"use_ad": True, "time_manager": time_manager})
    with pytest.raises(NotImplementedError) as excinfo:
        msg = "Time step cannot be adapted when the problem is linear."
        pp.run_time_dependent_model(model, model.params)
    assert msg in str(excinfo.value)


# -----> Testing adaptation after convergence for non-linear models
@pytest.mark.parametrize("model_class", non_linear_models)
@pytest.mark.parametrize("solution_type", ["parabolic", "trigonometric"])
def test_time_step_adaptation_after_convergence(model_class, solution_type):
    """Test iteration-based time step adaptation for non-linear models"""
    time_manager = pp.TimeManager(
        schedule=[0, 1],
        dt_init=0.3,
        dt_min_max=(0.1, 0.6),
        iter_max=10,
        iter_optimal_range=(4, 7),
        iter_relax_factors=(0.7, 1.3),
        recomp_factor=0.5,
        recomp_max=5,
        print_info=False,
    )
    params = {
        "use_ad": True,
        "time_manager": time_manager,
        "solution_type": solution_type,
        "num_cells": 5,
    }
    model = model_class(params=params)
    pp.run_time_dependent_model(model, model.params)
    np.testing.assert_equal(model.out["iterations"], [3, 3, 3])
    np.testing.assert_almost_equal(model.out["time_step"], [0.3, 0.39, 0.31], 6)


# -----> Testing adaptation after recomputation for non-linear models
tm_parabolic = pp.TimeManager(
    schedule=[0, 0.2],
    dt_init=0.18,
    dt_min_max=(0.05, 0.19),
    iter_optimal_range=(1, 2),
    iter_max=2,
)
params_parabolic = {
    "use_ad": True,
    "num_cells": 5,
    "time_manager": tm_parabolic,
    "solution_type": "parabolic",
    "max_iterations": 2,
    "nl_convergence_tol": 1e-8,
}

tm_trigonometric = pp.TimeManager(
    schedule=[0, 0.2],
    dt_init=0.18,
    dt_min_max=(0.04, 0.19),
    iter_optimal_range=(1, 2),
    iter_max=2,
)
params = {
    "use_ad": True,
    "num_cells": 5,
    "time_manager": tm_trigonometric,
    "solution_type": "trigonometric",
    "max_iterations": 2,
    "nl_convergence_tol": 1.5e-6,
}


@pytest.mark.parametrize("parameters", [params_parabolic])
def test_time_step_adapation_after_failure(request, parameters):
    """Test failed-to-convergence time step adaptation for non-linear models"""
    model = NonLinearSCF(params=parameters)
    pp.run_time_dependent_model(model, params=model.params)
    if "parabolic" in request.node.callspec.id:
        np.testing.assert_equal(model.out["iterations"], [2, 2, 2, 2])
        np.testing.assert_equal(model.out["recomputations"], [1, 1, 0, 0])
        np.testing.assert_almost_equal(sum(model.out["time_step"]), 0.2, 6)
        np.testing.assert_almost_equal(
            model.out["time_step"], [0.09, 0.05, 0.05, 0.01], 6
        )
    elif "trigonometric" in request.node.callspec.id:
        np.testing.assert_equal(model.out["iterations"], [2, 2, 2, 2, 2])
        np.testing.assert_equal(model.out["recomputations"], [2, 0, 0, 0, 0])
        np.testing.assert_almost_equal(sum(model.out["time_step"]), 0.2, 6)
        np.testing.assert_almost_equal(
            model.out["time_step"], [0.045, 0.04, 0.04, 0.04, 0.035], 6
        )
