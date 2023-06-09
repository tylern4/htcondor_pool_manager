
from typing import Dict, List

import pandas as pd
from slurm_cmds import SlurmCmdFailed
import math

from loguru import logger


class PoolManager:
    """Class representing a single Run"""

    def __init__(
        self,
        condor_provider=None,
        slurm_provider=None,
        configs=None,
        **kwargs,
    ):
        self.site = "nersc"
        self.condor_provider = condor_provider
        self.slurm_provider = slurm_provider
        self.configs = configs

    def get_current_slurm_workers(self) -> Dict:
        slurm_status = {}
        # Default slurm status to 0
        for compute_type in self.configs["compute_types"]:
            for _state in ["running", "pending"]:
                slurm_status[f"{compute_type}_{_state}"] = 0

        try:
            jobs = self.slurm_provider.squeue()
        except SlurmCmdFailed as err:
            logger.error(f"Slurm provider failed to run squeue, {err}")
            return slurm_status, pd.DataFrame([], columns=self.slurm_provider.columns)

        # Make dataframe to query with
        try:
            df = pd.DataFrame(jobs)
        except ValueError as err:
            logger.error(f"Pandas can't make df from outputs {jobs}, {err}")
            return slurm_status, pd.DataFrame([], columns=self.slurm_provider.columns)

        # Selections for running and pending
        mask_pending = (df["state"] == "PENDING")
        mask_running = (df["state"] == "RUNNING")

        # Selections for just condor jobs
        try:
            mask_condor = df["name"].str.contains("htcondor_worker")
        except AttributeError:
            logger.error(f"Problem with squeue output {df}")
            return slurm_status, pd.DataFrame([], columns=self.slurm_provider.columns)

        # For sites with multiple types we want to know
        for compute_type in self.configs["compute_types"]:
            mask_type = df["name"].str.contains(compute_type)
            # Each of these selects for a certian type of node based on a set of masks
            # Add the number of nodes to get how many are in each catogory
            slurm_status[f"{compute_type}_pending"] = sum(
                mask_type & mask_condor & mask_pending)
            slurm_status[f"{compute_type}_running"] = sum(
                mask_type & mask_condor & mask_running)

        slurm_running_df = df[mask_condor]
        logger.debug(f"Slurm status {slurm_status}")
        return slurm_status, slurm_running_df

    def get_condor_job_queue(self) -> List:
        condor_jobs = self.condor_provider.condor_q()
        # logger.debug(f"HTCondor job status {condor_jobs}")
        return condor_jobs

    def determine_condor_job_sizes(self, condor_jobs: Dict) -> Dict:
        # Brings dict output back into dataframe
        df = pd.DataFrame(condor_jobs)

        # Bins the jobs based on requested memory
        df["mem_bin"] = pd.cut(
            df["RequestMemory"],
            bins=self.configs["mem_bins"],
            labels=self.configs["labels"],
        )
        # Bins the jobs based on requested cpus
        df["cpu_bin"] = pd.cut(
            df["RequestCpus"],
            bins=self.configs["cpu_bins"],
            labels=self.configs["labels"],
        )

        condor_q_status = {}
        # Makes masks to get jobs based on status
        mask_running_status = df["JobStatus"].astype(int) == 2
        mask_idle_status = df["JobStatus"].astype(int) == 1
        mask_hold_status = df["JobStatus"].astype(int) == 5

        # If either cpu/mem are over limits then jobs will not run
        mask_over = df["cpu_bin"].str.contains(
            "over") | df["mem_bin"].str.contains("over")

        # Ignore anything on hold or over requesting resources
        condor_q_status["hold_and_impossible"] = sum(
            mask_over | mask_hold_status)

        # Check thorugh each of the compute types availible on the site
        for compute_type in self.configs["compute_types"]:
            mask_mem_type = df["mem_bin"].str.contains(f"{compute_type}")
            # mask_cpu_type = df["cpu_bin"].str.contains(f"{compute_type}")
            mask_type = (mask_mem_type & ~(mask_over | mask_hold_status))

            # Gets total number of each type that are idle or running
            condor_q_status[f"idle_{compute_type}"] = sum(
                mask_type & mask_idle_status)
            condor_q_status[f"running_{compute_type}"] = sum(
                mask_type & mask_running_status)

            # Gets total of resource requests of each type that are idle or running
            condor_q_status[f"{compute_type}_cpu_needed"] = sum(
                df[mask_type].RequestCpus)
            condor_q_status[f"{compute_type}_mem_needed"] = sum(
                df[mask_type].RequestMemory)
        logger.debug(f"condor_q_status {condor_q_status}")
        return condor_q_status

    def need_new_nodes(
        self,
        condor_job_queue: Dict,
        slurm_workers: Dict,
        machine_size: str,
    ) -> Dict:
        """
        Using the two dictionaries from the condor_q and squeue
        determine if we need any new workers for the machine_size types.
        """
        logger.debug(f"Checking if {machine_size} needs new workers")
        workers_needed = 0
        worker_sizes = self.configs["worker_sizes"]

        min_pool = self.configs["min_pool"].get(machine_size, 0)
        max_pool = self.configs["max_pool"].get(machine_size, 0)
        # Determines how many full (or partially full nodes) we need to create
        # [num cpus requested]/[number of cpus per node]
        cpu_nodes_needed = (
            condor_job_queue[f"{machine_size}_cpu_needed"] /
            worker_sizes[f"{machine_size}_cpu"]
        )
        # [mem requested]/[mem per node]
        mem_nodes_needed = (
            condor_job_queue[f"{machine_size}_mem_needed"] /
            worker_sizes[f"{machine_size}_mem"]
        )
        # Round the numbers up to be able to fill nodes
        cpu_nodes_needed = math.ceil(cpu_nodes_needed)
        mem_nodes_needed = math.ceil(mem_nodes_needed)

        # Get max of requests based on the number of nodes needed
        workers_needed += max(cpu_nodes_needed, mem_nodes_needed)

        # Total number running and pending  (i.e. total current worker pool)
        current_pool_size = (
            slurm_workers[f"{machine_size}_pending"]
            + slurm_workers[f"{machine_size}_running"]
        )

        # If workers_needed is higher than the pool we'll add the diference
        # Else we don't need workers (add 0)
        workers_needed = (workers_needed - current_pool_size)
        if workers_needed < 0:
            workers_needed = (abs(workers_needed) - current_pool_size)

        # If we have less running than the minimum we always need to add more
        # Either add what we need from queue (workers_needed)
        # Or what we're lacking in the pool (min - worker pool)
        if current_pool_size < min_pool:
            workers_needed = max(min_pool - current_pool_size, workers_needed)
            logger.info(f"{machine_size} workers_needd {workers_needed}")

        # Check to make sure we don't go over the max pool size
        if (workers_needed + current_pool_size) > max_pool:
            # Only add up to max pool and no more
            workers_needed = max_pool - current_pool_size

        # Makes sure we don't return a negative number
        workers_needed = workers_needed if workers_needed > 0 else 0
        logger.info(f"{machine_size} workers_needd {workers_needed}")
        return workers_needed if workers_needed > 0 else 0

    def need_cleanup(
        self,
        condor_job_queue: Dict,
        slurm_workers: Dict,
        machine_size: str,
    ) -> Dict:
        """
        Using the two dictionaries from the condor_q and squeue
        determine if we need any new workers for the machine_size types.
        """
        # Start off with minimum pool
        worker_sizes = self.configs["worker_sizes"]
        min_pool = self.configs["min_pool"][machine_size]

        # Determines how many full (or partially full nodes) we need to create
        _cpu = (
            condor_job_queue[f"{machine_size}_cpu_needed"] /
            worker_sizes[f"{machine_size}_cpu"]
        )
        _mem = (
            condor_job_queue[f"{machine_size}_mem_needed"] /
            worker_sizes[f"{machine_size}_mem"]
        )

        # Round the numbers up
        _cpu = math.ceil(_cpu)
        _mem = math.ceil(_mem)
        workers_needed = max(min_pool, max(_cpu, _mem))
        logger.debug(f"{machine_size} => {workers_needed}")

        # Total number running and pending to run (i.e. worker pool)
        current_pool_size = (
            slurm_workers[f"{machine_size}_pending"]
            + slurm_workers[f"{machine_size}_running"]
        )

        # If workers_needed is higher than the pool we'll add the diference
        # Else we don't need workers (add 0)
        workers_needed = (workers_needed - current_pool_size)

        if workers_needed >= min_pool:
            workers_needed = workers_needed - min_pool

        workers_needed = workers_needed if workers_needed < 0 else 0
        logger.info(f"{machine_size} need cleanup {workers_needed}")
        return workers_needed

    def run_cleanup(
        self,
        slurm_running_df,
        cleanup_num: int,
        compute_type: str,
    ):
        # Runs a condor_q autoformat to get the desired columns back

        try:
            logger.debug(slurm_running_df.shape)
        except NameError as err:
            logger.debug(f"No slurm nodes yet, {err}")
            return None

        slurm_running_df.sort_values("TIME_SEC", inplace=True, ascending=False)
        # Any pending nodes are at the front of the list to remove
        # Gets the list of jobs IDs that are pending
        pending = slurm_running_df[slurm_running_df["state"] == "PENDING"].jobid

        num = 0
        # Loops over the pending job ids
        for pend in pending:
            logger.debug(f"pending job {pend}")
            # Check to see if we still need to cleanup jobs
            if num >= cleanup_num:
                return True
            try:
                # Makes sure the job id is an int
                job_id = int(pend)
                logger.info(f"JobID is {job_id}")
            except (IndexError, TypeError, AttributeError):
                # Continue to next jobid if there is a problem
                continue
            num += 1
            logger.info(f"Removing pending JobID {job_id}")
            self.slurm_provider.scancel(
                job_id=job_id, cluster=self.configs["clusters"][compute_type])

        # Gets the idle nodes from condor
        idle_nodes = self.condor_provider.condor_idle()
        logger.debug(f"Idle nodes {idle_nodes}")

        if num >= cleanup_num or len(idle_nodes) == 0:
            return True

        for node in idle_nodes:
            if num >= cleanup_num:
                return True
            try:
                node_mask = (slurm_running_df["nodelist"] == node)
                type_mask = slurm_running_df["name"].str.contains(compute_type)
                job_id = slurm_running_df[node_mask & type_mask]
                job_id = int(job_id['jobid'])
                logger.debug(f"JobID is {job_id}")
            except IndexError as err:
                logger.warning(err)
                continue
            except TypeError as err:
                logger.warning(err)
                continue
            except AttributeError as err:
                logger.warning(err)
                continue
            num += 1
            logger.info(f"Removing idle node {node} with JobID {job_id}")
            self.slurm_provider.scancel(
                job_id=job_id, cluster=self.configs["clusters"][compute_type])

        return None

    def run_sbatch(self, new_workers: int = 0, compute_type: str = "medium"):
        jobs = []
        for _ in range(new_workers):
            status = self.slurm_provider.sbatch(
                compute_type=compute_type,
                cluster=self.configs["clusters"][compute_type],
            )
            jobs.append(status["stdout"])
            logger.info(f"Started new job with id: {status['stdout']}")

        return jobs
