import contextlib
import copy
import cProfile
import io
import pstats
import time
import traceback
from pstats import SortKey
from typing import List, cast

import joblib
from joblib import Parallel, delayed

from sperl.config.general import GeneralConfig
from sperl.config.single_run import SingleRunConfig
from sperl.generatedata.measures.measure_statistics import MeasureStatistics
from sperl.generatedata.measures.stepwise_occupancy import StepwiseOccupancy
from sperl.generatedata.meta_trajectory import MetaTrajectoryGenerator


class DataGenerator:
    def __init__(self, run_config: SingleRunConfig, general_config: GeneralConfig):
        self.run_config = run_config
        self.general_config = general_config
        self.distance_statistics: MeasureStatistics
        self.num_exceptions: int
        self._traj_generators: List[MetaTrajectoryGenerator] = self._create_generators()

    def _create_generators(self) -> List[MetaTrajectoryGenerator]:
        seed_sequences = self.run_config.seed_seq.spawn(
            self.general_config.num_meta_trajectories
        )
        return [
            MetaTrajectoryGenerator(self.run_config, self.general_config, s)
            for s in seed_sequences
        ]

    def run(self):
        distances, self.num_exceptions = self._distances_num_exceptions()
        distances = copy.deepcopy(distances)
        self.distance_statistics = MeasureStatistics(distances)

    @property
    def finished(self):
        return self._traj_generators[0].finished

    @property
    def num_steps(self):
        return self._traj_generators[0].step

    def _distances_num_exceptions(self):
        tasks = [
            delayed(self._run_traj_without_profiler)(g) for g in self._traj_generators
        ]
        if self.general_config.profile:
            tasks[0] = delayed(self._run_with_profiler)(self._traj_generators[0])
        distances_n_status = Parallel(n_jobs=self.general_config.num_jobs)(tasks)

        distances_n_status = cast(List, distances_n_status)
        self._traj_generators = [d[0] for d in distances_n_status if d[1]]
        distances = [g.distances for g in self._traj_generators]
        num_exceptions = self.general_config.num_meta_trajectories - len(distances)
        return distances, num_exceptions

    def _print_diagnostic_info(self):
        print("_distances_num_exceptions")
        self.run_config.print_info()
        print(
            "  num meta trajectories: ",
            self.general_config.num_meta_trajectories,
        )

    def _run_with_profiler(self, generator: MetaTrajectoryGenerator):
        with cProfile.Profile() as pr:
            start = time.time()
            try:
                generator.run(self.general_config.store_steps)
            finally:
                end = time.time()
                num_updates = len(
                    generator.distances.distance_dict[
                        StepwiseOccupancy.name()
                    ].measures_list
                )
                runtime = end - start
                print("=" * 120, "\n", "=" * 120)
                print("num_updates = ", num_updates)
                print("runtime = ", runtime)
                print("num_updates / second = ", num_updates / runtime)
                print("=" * 120, "\n", "=" * 120)
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
                ps.print_stats(0.1)
                print(s.getvalue())

        return generator, True

    def _run_traj_without_profiler(self, generator: MetaTrajectoryGenerator):
        try:
            generator.run(self.general_config.store_steps)
        except Exception:
            traceback.print_exc()
            print("*" * 90)
            return None, False
        return generator, True

    # the following function is taken from
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    @contextlib.contextmanager
    def _tqdm_joblib(self, tqdm_object):
        """
        Context manager to patch joblib to report into tqdm progress bar given as
        argument
        """

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
