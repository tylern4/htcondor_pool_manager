"""
JAWS Daemon process periodically checks on runs and performs actions to usher
them to the next state.
"""

from pathlib import Path
import sys
from pool_manager import PoolManager
from htcondor_cmds import HTCondor
from slurm_cmds import Slurm
import schedule
import time
import logging
import json


logger = logging.getLogger(__package__)


def load_configs(configs):
    # Used to setup cpu and mem bins for splitting
    configs["cpu_bins"] = [0]
    configs["mem_bins"] = [0]
    configs["labels"] = []
    # Loop over compute types and build the rest of the configuration needed by pool manager
    for compute_type in configs["compute_types"]:
        configs["labels"].append(compute_type)
        for cpu_mem in ["cpu", "mem"]:
            configs[f"{cpu_mem}_bins"].append(
                configs["worker_sizes"][f"{compute_type}_{cpu_mem}"]
            )
    # Puts max limits for sizes of pools in the binning
    configs["cpu_bins"].append(10_000)
    configs["mem_bins"].append(10_000)
    configs["labels"].append("over")

    return configs


class PoolManagerDaemon:
    """
    Daemon that periodically checks on this site's active runs.
    It prompts active runs to query Cromwell or Globus, as appropriate.
    When Globus uploads are completed, the run is submitted to Cromwell.
    When Cromwell execution has completed, the output is downloaded using Globus.
    When the download is finished, the run is complete.
    """

    def __init__(self, conf):
        logger.info("Initializing pool manager daemon")

        self.time_add_worker_pool = int(conf["time_add_worker_pool"])
        self.time_rm_worker_pool = int(conf["time_rm_worker_pool"])
        self.compute_types = conf["compute_types"]
        self.wanted_columns = conf["wanted_columns"]
        self.user_name = conf["user_name"]
        self.squeue_args = conf["squeue_args"]
        self.script_path = Path.cwd()

        self.configs = load_configs(conf)

    def start_daemon(self):
        """
        Run scheduled task(s) periodically.
        """
        schedule.every(self.time_add_worker_pool).seconds.do(
            self.add_worker_pool)
        schedule.every(self.time_rm_worker_pool).seconds.do(
            self.rm_worker_pool)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def add_worker_pool(self):
        """
        Check for runs in particular states.
        """
        pool = PoolManager(
            condor_provider=HTCondor(columns=self.wanted_columns),
            slurm_provider=Slurm(
                user_name=self.user_name,
                extra_args=self.squeue_args,
                script_path=self.script_path,
            ),
            configs=self.configs,
        )
        slurm_status, slurm_running_df = pool.get_current_slurm_workers()
        condor_status = pool.get_condor_job_queue()
        work_status = pool.determine_condor_job_sizes(condor_status)
        for compute_type in self.compute_types:
            new_workers = pool.need_new_nodes(
                work_status, slurm_status, compute_type)
            logger.info(
                f"Looking to add {abs(new_workers)} for {compute_type} pool")
            if new_workers > 0:
                pool.run_sbatch(abs(new_workers), compute_type)

    def rm_worker_pool(self):
        """
        Check for runs in particular states.
        """
        pool = PoolManager(
            condor_provider=HTCondor(columns=self.wanted_columns),
            slurm_provider=Slurm(
                user_name=self.user_name,
                extra_args=self.squeue_args,
                script_path=self.script_path,
            ),
            configs=self.configs,
        )
        slurm_status, slurm_running_df = pool.get_current_slurm_workers()
        condor_status = pool.get_condor_job_queue()
        work_status = pool.determine_condor_job_sizes(condor_status)
        for compute_type in self.compute_types:
            old_workers = pool.need_cleanup(
                work_status, slurm_status, compute_type)
            logger.info(
                f"Looking to remove {abs(old_workers)} from {compute_type} pool"
            )
            if old_workers < 0:
                pool.run_cleanup(slurm_running_df, abs(
                    old_workers), compute_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sys.argv[1]
    with open(sys.argv[1]) as f:
        conf = json.load(f)

    pool = PoolManagerDaemon(conf=conf)
    pool.start_daemon()
