import fcntl
import os
import pickle
import sys
import time
from typing import List, cast

from sperl import utils
from sperl.config.config_manager import ConfigManager
from sperl.config.single_run import SingleRunConfig
from sperl.generatedata.data_generator import DataGenerator


class ExperimentManager:
    def __init__(self, config: ConfigManager):
        self.config: ConfigManager = config
        self._dir_name = self._create_unique_directory()
        self._store_arguments()
        self.start_time: float

    def run(self):
        self.start_time = time.time()
        runtime = utils.get_runtime(self._run)
        print("Elapsed time: {}".format(runtime))
        timefile = os.path.join(self._dir_name, "time.txt")
        with open(timefile, "w") as f:
            f.write(str(runtime))

    def _run(self):
        utils.set_random_seed(self.config.experiment.seed)
        run_confs = self.config.get_single_run_configs()
        num_exceptions = [self._data_single_run(cg) for cg in run_confs]
        print("Total num exceptions = {}".format(sum(cast(List, num_exceptions))))

    def _data_single_run(self, run_config: SingleRunConfig) -> int:
        g = DataGenerator(run_config, self.config.general)
        while not g.finished:
            g.run()
            print("storing intermediate result")
            self._store_single_run(g, run_config)
        print("Num exceptions: {}".format(g.num_exceptions))
        return g.num_exceptions

    def _store_arguments(self):
        args = "args " + " ".join(sys.argv[1:])
        fn = os.path.join(self._dir_name, "args.txt")
        with open(fn, "x") as f:
            f.write(args)

    def _create_unique_directory(self):
        counter = 0
        os.makedirs(self.config.general.base_path, exist_ok=True)
        lockfile_path = os.path.join(
            self.config.general.base_path, "directory_creation.lock"
        )
        with open(lockfile_path, "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)

            while True:
                new_directory = os.path.join(
                    self.config.general.base_path, f"exp_{counter:03d}"
                )

                if not os.path.exists(new_directory):
                    os.mkdir(new_directory)
                    break

                counter += 1
            fcntl.flock(lockfile, fcntl.LOCK_UN)
        return new_directory

    def _store_single_run(
        self,
        generator: DataGenerator,
        run_config: SingleRunConfig,
    ):
        os.makedirs(self._dir_name, exist_ok=True)
        basepath = os.path.join(self._dir_name, run_config.get_str(generator.num_steps))
        filepath = basepath + ".pickle"
        with open(filepath, "wb") as f:
            pickle.dump(
                (
                    self.config.general,
                    run_config,
                    generator.distance_statistics,
                ),
                f,
            )

        timepath = basepath + "_time.txt"
        with open(timepath, "w") as f:
            runtime = str(time.time() - self.start_time)
            f.write(runtime)
