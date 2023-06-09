import json
from loguru import logger
import time
import pandas as pd
from slurm_cmds import Slurm, slurm_time_to_sec


class SFapiError(Exception):
    pass


class SFapiCmdFailed(SFapiError):
    pass


try:
    from sfapi_client import Client
    from sfapi_client.compute import Compute, Machine
    from sfapi_client.jobs import JobSacct, JobSqueue, JobCommand
except ImportError:
    logger.error("Could not load sfapi_client")


class SFapiSlurm(Slurm):
    all_slurm_keys = ['account', 'tres_per_node', 'min_cpus', 'min_tmp_disk',
                      'end_time', 'features', 'group', 'over_subscribe', 'jobid', 'name',
                      'comment', 'time_limit', 'min_memory', 'req_nodes', 'command', 'priority',
                      'qos', 'reason', '', 'st', 'user', 'reservation', 'wckey', 'exc_nodes', 'nice',
                      's:c:t', 'exec_host', 'cpus', 'nodes', 'dependency', 'array_job_id',
                      'sockets_per_node', 'cores_per_socket', 'threads_per_core', 'array_task_id',
                      'time_left', 'time', 'nodelist', 'contiguous', 'partition', 'nodelist(reason)',
                      'start_time', 'state', 'uid', 'submit_time', 'licenses', 'core_spec',
                      'schednodes', 'work_dir'
                      ]

    def __init__(
        self,
        user_name: str = "tylern",
        site: str = "perlmutter",
        extra_args: str = None,
        script_path: str = "",
        **kwargs,
    ):
        self.user_name = user_name
        self.site = site
        self.script_path = script_path
        self.client = Client()
        self.sfapi = self.client.compute(Machine.perlmutter)

    @property
    def squeue_columns(self):
        return self.all_slurm_keys

    def squeue(self) -> dict:
        """
        Runs the squeue command and returns a dictionary
        compatible with a pandas dataframe
        """
        try:
            jobs = self.sfapi.jobs(user=self.user_name)
        except Exception as err:
            logger.debug(f"{err}")
            df = pd.DataFrame(columns=self.all_slurm_keys)
            return df.to_dict()

        if len(jobs) == 0:
            df = pd.DataFrame(columns=self.all_slurm_keys)
            return df.to_dict()

        df = pd.DataFrame([j.dict() for j in jobs])
        # Drops rows if they have nan values
        # df = df.dropna(axis=0)

        # Adds a new column of time in seconds left to run
        df["TIME_SEC"] = df["time_left"].apply(slurm_time_to_sec)
        return df.to_dict()

    def scancel(self, job_id: int = 0, cluster: str = ""):
        logger.info(f"Removing {job_id}")
        job = self.sfapi.job(jobid=job_id, command=JobCommand.sacct)
        return job.cancel()

    def sbatch(self, compute_type: str = "medium", cluster: str = "", wait_time: int = 10):

        try:
            job = self.sfapi.submit_job(f"{self.script_path}/htcondor_worker_{compute_type}.job")
        except SFapiCmdFailed:
            logger.error("sbatch failed")
            return {"stdout": "failed", "stderr": "failed", "returncode": 1}

        return {"stdout": job.jobid, "stderr": "", "returncode": 0}
