import json
import logging
import time
import pandas as pd
from slurm_cmds import Slurm, slurm_time_to_sec

logger = logging.getLogger(__package__)


class SFapiError(Exception):
    pass


class SFapiCmdFailed(SFapiError):
    pass


try:
    from SuperfacilityAPI import SuperfacilityAPI, SuperfacilityAccessToken, error_warnings
except ImportError:
    logging.error("Could not load SuperfacilityAPI")


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
        self.sfapi_token = SuperfacilityAccessToken()
        self.sfapi = SuperfacilityAPI(token=self.sfapi_token.token)

    def squeue(self) -> dict:
        """
        Runs the squeue command and returns a dictionary
        compatible with a pandas dataframe
        """
        try:
            jobs = self.sfapi.get_jobs(
                user=self.user_name,
                site=self.site,
                sacct=False
            )
        except error_warnings.SuperfacilitySiteDown as err:
            logger.debug(f"{err}")
            df = pd.DataFrame(columns=self.all_slurm_keys)
            return df.to_dict()

        if len(jobs['output']) == 0:
            df = pd.DataFrame(columns=self.all_slurm_keys)
            return df.to_dict()

        df = pd.DataFrame.from_records(jobs['output'])

        # Drops rows if they have nan values
        df = df.dropna(axis=0)

        # Adds a new column of time in seconds left to run
        df["time_sec"] = df["time_left"].apply(slurm_time_to_sec)

        return df.to_dict()

    def scancel(self, job_id: int = 0, cluster: str = ""):

        cluster = "" if cluster is None else cluster
        if cluster != "":
            raise SFapiCmdFailed("Cluster not supported with sfapi")

        return self.sfapi.delete_job(site=self.site, jobid=job_id)

    def sbatch(self, compute_type: str = "medium", cluster: str = "", wait_time: int = 10):

        cluster = "" if cluster is None else cluster
        if cluster != "":
            raise SFapiCmdFailed("Cluster not supported with sfapi")

        try:
            ret = self.sfapi.post_job(site=self.site,
                                      script=f"{self.script_path}/htcondor_worker_{compute_type}.job",
                                      isPath=True,
                                      run_async=True
                                      )
        except SFapiCmdFailed:
            logging.error("sbatch failed")
            return {"stdout": "failed", "stderr": "failed", "returncode": 1}
        except SuperfacilityAPI.SuperfacilitySiteDown:
            return {"stdout": "failed", "stderr": "failed", "returncode": 1}

        for i in range(wait_time):
            time.sleep(1)
            task_info = self.sfapi.tasks(task_id=ret['task_id'])
            if task_info['result'] != None:
                task = json.loads(task_info['result'])
                return {"stdout": int(task['jobid']), "stderr": str(task['error']), "returncode": 0}
            logging.debug(f"Trying {i+1}")

        return {"stdout": {'task_id': int(ret['task_id'])}, "stderr": ret['error'], "returncode": 0}


if __name__ == '__main__':
    import sys
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    slurm = SFapiSlurm(
        script_path='/global/homes/t/tylern/htcondor_workflow_scron',
        user_name='nmjaws',
        site='cori'
    )

    # job = slurm.sbatch(compute_type='large')
    # print(job)
    # job = slurm.sbatch(compute_type='gpu')
    # print(job)

    squeue = slurm.squeue()

    jobs = pd.DataFrame.from_dict(squeue)
    logging.debug("Wating 1s...")
    time.sleep(1)

    to_cancel = {'jobid': []}
    if len(jobs) > 0:
        to_cancel = jobs[jobs['name'].str.contains('htcondor_worker')]

    for job_id in to_cancel['jobid']:
        # ret = slurm.scancel(job_id=job_id)
        logging.debug(json.dumps(job_id))

    wait_time = 3
    for remaining in range(wait_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(remaining))
        sys.stdout.flush()
        time.sleep(1)

    squeue = slurm.squeue()
    print(json.dumps(squeue))
