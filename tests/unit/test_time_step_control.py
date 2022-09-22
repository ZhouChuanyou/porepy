import pytest
import numpy as np

import porepy as pp
from porepy.numerics.time_step_control import TimeSteppingControl as Ts


class TestParameterInputs:
    """The following tests are written to check the sanity of the input parameters"""

    # TODO: Remove parametrization
    # TODO: Purge dt_min_max from passed parameters
    # TODO: Check for default values of dt_min_max
    @pytest.mark.parametrize("constant_dt", [True, False])
    def test_default_parameters_and_attribute_initialization(self, constant_dt):
        """Test the default parameters and initialization of attributes."""
        tsc = Ts(
            schedule=[0, 1], dt_init=0.2, constant_dt=constant_dt, dt_min_max=(0.1, 0.5)
        )
        np.testing.assert_equal(tsc.schedule, np.array([0, 1]))
        assert tsc.time_init == 0
        assert tsc.time_final == 1
        assert tsc.dt_init == 0.2
        assert tsc.dt_min_max == (0.1, 0.5)
        assert tsc.iter_max == 15
        assert tsc.iter_optimal_range == (4, 7)
        assert tsc.iter_relax_factors == (0.7, 1.3)
        assert tsc.recomp_factor == 0.5
        assert tsc.recomp_max == 10
        assert tsc.time == 0
        assert tsc.dt == 0.2

    # def test_initialization_with_constant_dt(self):
    #     """Test the proper initialization of parameters when setting a constant time step."""
    #     tsc = Ts(schedule=[0,1], dt_init=1., constant_dt=True)

    #     assert tsc.dt == 1
    #     assert tsc.dt_min == 1
    #     assert tsc.dt_max == 1
    #     assert tsc.iter_low_factor == 1
    #     assert tsc.iter_upp_factor == 1
    #     assert tsc.recomp_factor == 1
    #     assert tsc.is_constant

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([], 0.1, (0.1, 1)),
            ([1], 0.1, (0.1, 1)),
        ],
    )
    def test_schedule_length_greater_than_2(self, schedule, dt_init, dt_min_max):
        """An error should be raised if len(schedule) < 2."""
        msg = "Expected schedule with at least two elements."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([-1, 10], 0.1, (0.1, 1)),
            ([0, -3], 0.1, (0.1, 1)),
            ([1, 2, -100, 3, 4], 0.1, (0.1, 1)),
        ],
    )
    def test_positive_time_in_schedule(self, schedule, dt_init, dt_min_max):
        """An error should be raised if a negative time is encountered in the schedule."""
        msg = "Encountered at least one negative time in schedule."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([0, 1, 2, 5, 1], 0.1, (0.1, 1)),
            ([0, 1, 1, 2], 0.1, (0.1, 1)),
            ([100, 200, 50], 0.1, (0.1, 1)),
        ],
    )
    def test_strictly_increasing_time_in_schedule(self, schedule, dt_init, dt_min_max):
        """An error should be raised if times in schedule are not strictly increasing."""
        msg = "Schedule must contain strictly increasing times."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "schedule, dt_init",
        [
            ([0, 1], 0.4),
            ([0, 1], 0.3333),
            ([0, 0.4, 0.5, 0.8, 1], 0.2),
        ],
    )
    def test_schedule_matches_constant_time_step(self, schedule, dt_init):
        """An error should be raised if the schedule does not match the constant time step."""
        msg = "Mismatch between the time step and scheduled time."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, constant_dt=True)
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max",
        [
            ([0, 1], -1, (0.1, 1)),
            ([0, 1], 0, (0.1, 1)),
        ],
    )
    def test_positive_initial_time_step(self, schedule, dt_init, dt_min_max):
        """An error should be raised if initial time step is non-positive."""
        msg = "Initial time step must be positive."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=schedule, dt_init=dt_init, dt_min_max=dt_min_max)
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_initial_time_step_smaller_than_final_time(self):
        """An error should be raised if initial time step is larger than final time."""
        msg = "Initial time step cannot be larger than final simulation time."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=1.0001, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_initial_time_step_larger_than_minimum_time_step(self):
        """An error should be raised if initial time step is less than minimum time step."""
        msg = "Initial time step cannot be smaller than minimum time step."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.09, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_initial_time_step_smaller_than_maximum_time_step(self):
        """An error should be raised if initial time step is larger than the maximum time
        step."""
        msg = "Initial time step cannot be larger than maximum time step."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.51, dt_min_max=(0.1, 0.5))
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("iter_max", [0, -1])
    def test_max_number_of_iterations_positive(self, iter_max):
        """An error should be raised if the maximum number of iterations is not positive."""
        msg = "Maximum number of iterations must be positive."
        with pytest.raises(ValueError) as excinfo:
            Ts(schedule=[0, 1], dt_init=0.1, dt_min_max=(0.1, 0.5), iter_max=iter_max)
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_lower_iter_endpoint_smaller_than_upper_iter_endpoint(self):
        """An error should be raised if the lower endpoint of the optimal iteration range is
        larger than the upper endpoint of the optimal iteration range."""
        iter_optimal_range = (3, 2)
        msg = (
            f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration range "
            f"cannot be larger than upper endpoint '{iter_optimal_range[1]}'."
        )
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_max=5,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_upper_iter_endpoint_less_or_equal_than_max_iter(self):
        """An error should be raised if the upper endpoint of the optimal iteration range is
        larger than the maximum number of iterations."""
        iter_optimal_range = (2, 6)
        iter_max = 5
        msg = (
            f"Upper endpoint '{iter_optimal_range[1]}' of optimal iteration range "
            f"cannot be larger than maximum number of iterations '{iter_max}'."
        )
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_max=iter_max,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_lower_iter_endpoint_greater_than_or_equal_to_zero(self):
        """An error should be raised if the lower iteration range is less than zero."""
        iter_optimal_range = (-1, 2)
        msg = (
            f"Lower endpoint '{iter_optimal_range[0]}' of optimal iteration range "
            "cannot be negative."
        )
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_max=5,
                iter_optimal_range=iter_optimal_range,
            )
        assert msg in str(excinfo.value)

    # TODO: Remove dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, iter_relax_factors",
        [
            ([0, 1], 0.1, (0.1, 0.5), (1.0, 1.3)),
            ([0, 1], 0.1, (0.1, 0.5), (1.05, 1.3)),
        ],
    )
    def test_under_relaxation_factor_less_than_one(
        self, schedule, dt_init, dt_min_max, iter_relax_factors
    ):
        """An error should be raised if under-relaxation factor >= 1"""
        msg = "Expected under-relaxation factor < 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule,
                dt_init,
                dt_min_max=dt_min_max,
                iter_relax_factors=iter_relax_factors,
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, iter_relax_factors",
        [
            ([0, 1], 0.1, (0.1, 0.5), (0.7, 1.0)),
            ([0, 1], 0.1, (0.1, 0.5), (0.7, 0.95)),
        ],
    )
    def test_over_relaxation_factor_greater_than_one(
        self, schedule, dt_init, dt_min_max, iter_relax_factors
    ):
        """An error should be raised if over-relaxation factor <= 1"""
        msg = "Expected over-relaxation factor > 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule,
                dt_init,
                dt_min_max=dt_min_max,
                iter_relax_factors=iter_relax_factors,
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_dt_min_times_over_relax_factor_less_than_dt_max(self):
        """An error should be raised if dt_min * over_relax_factor > dt_max."""
        msg_dtmin_over = "Encountered dt_min * over_relax_factor > dt_max. "
        msg_osc = (
            "The algorithm will behave erratically for such a combination of parameters. "
            "See documentation of `dt_min_max` or `iter_relax_factors`."
        )
        msg = msg_dtmin_over + msg_osc
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_relax_factors=(0.9, 6),
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    def test_dt_max_times_under_relax_factor_greater_than_dt_min(self):
        """An error should be raised if dt_max * under_relax_factor < dt_min."""
        msg_dtmax_under = "Encountered dt_max * under_relax_factor < dt_min. "
        msg_osc = (
            "The algorithm will behave erratically for such a combination of parameters. "
            "See documentation of `dt_min_max` or `iter_relax_factors`."
        )
        msg = msg_dtmax_under + msg_osc
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=[0, 1],
                dt_init=0.1,
                dt_min_max=(0.1, 0.5),
                iter_relax_factors=(0.01, 1.3),
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, recomp_factor",
        [
            ([0, 1], 0.1, (0.1, 0.5), 1),
            ([0, 1], 0.1, (0.1, 0.5), 1.05),
        ],
    )
    def test_recomputation_factor_less_than_one(
        self, schedule, dt_init, dt_min_max, recomp_factor
    ):
        """An error should be raised if the recomputation factor is greater or equal to one."""
        msg = "Expected recomputation factor < 1."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=schedule,
                dt_init=dt_init,
                dt_min_max=dt_min_max,
                recomp_factor=recomp_factor,
            )
        assert msg in str(excinfo.value)

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule, dt_init, dt_min_max, recomp_max",
        [
            ([0, 1], 0.1, (0.1, 0.5), -1),
            ([0, 1], 0.1, (0.1, 0.5), 0),
        ],
    )
    def test_number_of_recomp_attempts_greater_than_zero(
        self, schedule, dt_init, dt_min_max, recomp_max
    ):
        """An error should be raised if the number of recomputation attempts is zero or
        negative."""
        msg = "Number of recomputation attempts must be > 0."
        with pytest.raises(ValueError) as excinfo:
            Ts(
                schedule=schedule,
                dt_init=dt_init,
                dt_min_max=dt_min_max,
                recomp_max=recomp_max,
            )
        assert msg in str(excinfo.value)


class TestTimeControl:
    """The following tests are written to check the overall behavior of the time-stepping
    algorithm"""

    # TODO: Purge dt_min_max from parameters
    def test_final_simulation_time(self):
        """Test if final simulation time returns None, irrespectively of parameters
        passed in next_time_step()."""
        # Assume we reach the final time
        tsc = Ts(schedule=[0, 1], dt_init=0.1, dt_min_max=(0.1, 0.5))
        tsc.time = 1
        dt = tsc.next_time_step(iterations=1000, recompute_solution=True)
        assert dt is None
        # Now, assume we are above the final time
        tsc = Ts(schedule=[0, 1], dt_init=0.1, dt_min_max=(0.1, 0.5))
        tsc.time = 2
        dt = tsc.next_time_step(iterations=0, recompute_solution=False)
        assert dt is None

    # TODO: Purge dt_min_max from parameters
    def test_constant_time_step(self):
        """Test if a constant time step is returned, independent of any configuration or
        input."""
        tsc = Ts(schedule=[0, 1], dt_init=0.1, constant_dt=True)

        dt = tsc.next_time_step(1000, True)
        assert dt == 0.1
        dt = tsc.next_time_step(1000, False)
        assert dt == 0.1
        dt = tsc.next_time_step(1, True)
        assert dt == 0.1
        dt = tsc.next_time_step(1, False)
        assert dt == 0.1

    # TODO: Purge dt_min_max from parameters
    def test_non_recomputed_solution_conditions(self):
        """Test behaviour of the algorithm when the solution should NOT be recomputed"""
        # Check if internal flag _recomp_sol remains unchanged when recompute_solution=False
        # regardless of the number of iterations provided by the user
        tsc = Ts([0, 1], 0.1, dt_min_max=(0.1, 0.5))
        tsc.next_time_step(iterations=5)
        assert not tsc._recomp_sol
        tsc.next_time_step(iterations=1000)
        assert not tsc._recomp_sol
        # Check if _recomp_num resets to zero when solution is NOT recomputed
        tsc = Ts([0, 1], 0.1, dt_min_max=(0.1, 0.5))
        tsc._recomp_num = 3  # manually change recomputation attempts
        tsc.next_time_step(iterations=5)
        assert tsc._recomp_num == 0
        # Assume recompute_solution=True, but we reach or exceeded maximum number of attempts
        tsc = Ts([0, 1], 0.1, dt_min_max=(0.1, 0.5), recomp_max=5)
        tsc._recomp_num = 5
        with pytest.raises(ValueError) as excinfo:
            msg = (
                f"Solution did not converge after {tsc.recomp_max} recomputing "
                "attempts."
            )
            tsc.next_time_step(iterations=5, recompute_solution=True)
        assert tsc._recomp_sol and (msg in str(excinfo.value))

    # TODO: Purge dt_min_max from parameters
    def test_recomputed_solutions(self):
        """Test behaviour of the algorithm when the solution should be recomputed. Note
        that this should be independent of the number of iterations that the user passes"""
        tsc = Ts([0, 100], 2, dt_min_max=(0.1, 10), recomp_factor=0.5)
        tsc.time = 5
        tsc.dt = 1
        tsc.next_time_step(recompute_solution=True, iterations=1000)
        # We expect the time step to be reduced half, time to be corrected (decreased
        # accordingly), _recomp_sol == True, and the counter _recomp_num increased by 1.
        assert tsc.time == 4
        assert tsc.dt == 0.5
        assert tsc._recomp_sol
        assert tsc._recomp_num == 1

    # TODO: Purge dt_min_max from parameters
    def test_recomputed_solution_with_calculated_dt_less_than_dt_min(self):
        """Test when a solution is recomputed and the calculated time step is less than
        the minimum allowable time step, the time step is indeed the minimum time step"""
        tsc = Ts([0, 100], 2, dt_min_max=(0.6, 10), recomp_factor=0.5)
        # Emulate the scenario where the solution must be recomputed
        tsc.time = 5
        tsc.dt = 1
        tsc.next_time_step(iterations=1000, recompute_solution=True)
        # First the algorithm will reduce dt by half (so dt=0.5), but this is less than
        # dt_min. Hence, dt_min should be set.
        assert tsc.dt == tsc.dt_min_max[0]

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("iterations", [1, 3, 5])
    def test_decreasing_time_step(self, iterations):
        """Test if the time step decreases after the number of iterations is less or equal
        than the lower endpoint of the optimal iteration range by its corresponding factor"""
        tsc = Ts(
            [0, 100],
            2,
            dt_min_max=(0.1, 10),
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        tsc.dt = 1
        tsc.next_time_step(iterations=iterations, recompute_solution=False)
        assert tsc.dt == 1.3

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("iterations", [9, 11, 13])
    def test_increasing_time_step(self, iterations):
        """Test if the time step is restricted after the number of iterations is greater or
        equal than the upper endpoint of the optimal iteration range by its corresponding
        factor"""
        tsc = Ts(
            [0, 100],
            2,
            dt_min_max=(0.1, 10),
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        tsc.dt = 1
        tsc.next_time_step(iterations=iterations, recompute_solution=False)
        assert tsc.dt == 0.7

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("iterations", [6, 7, 8])
    def test_time_step_within_optimal_iteration_range(self, iterations):
        """Test if the time step remains unchanged when the number of iterations lies
        between the optimal iteration range"""
        tsc = Ts(
            [0, 100],
            2,
            dt_min_max=(0.1, 10),
            iter_optimal_range=(5, 9),
            iter_relax_factors=(0.7, 1.3),
        )
        tsc.dt = 1
        tsc.next_time_step(iterations=iterations)
        assert tsc.dt == 1

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("dt", [0.13, 0.1, 0.075])
    def test_time_step_less_than_dt_min(self, dt):
        """Test if the algorithm passes dt_min when the calculated dt is less than dt_min"""
        tsc = Ts([0, 100], 2, dt_min_max=(0.1, 10), iter_optimal_range=(4, 7))
        tsc.dt = dt
        tsc.next_time_step(iterations=7)
        assert tsc.dt == tsc.dt_min_max[0]

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize("dt", [9, 10, 15])
    def test_time_step_greater_than_dt_max(self, dt):
        """Test if the algorithm passes dt_max when the calculated dt is greater than dt_max"""
        tsc = Ts([0, 100], 2, dt_min_max=(0.1, 10), iter_optimal_range=(4, 7))
        tsc.dt = dt
        tsc.next_time_step(iterations=4)
        assert tsc.dt == tsc.dt_min_max[1]

    # TODO: Purge dt_min_max from parameters
    @pytest.mark.parametrize(
        "schedule",
        [
            [0, 1],
            [0, 10, 20, 30],
            [10, 11, 15, 16, 19, 20],
            [0, 0.01, 1 * pp.HOUR, 2 * pp.HOUR, 100 * pp.HOUR, 101 * pp.HOUR],
        ],
    )
    def test_hitting_schedule_times(self, schedule):
        """Test if algorithm respects the passed target times from the schedule"""
        tsc = Ts(schedule, 0.1, dt_min_max=(0.01, 0.1 * schedule[-1]))
        for time in schedule[1:]:
            tsc.time = 0.99 * time
            tsc.dt = tsc.dt_min_max[1]
            tsc.next_time_step(recompute_solution=False, iterations=4)
            assert time == tsc.time + tsc.dt
