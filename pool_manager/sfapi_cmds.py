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
    from SuperfacilityAPI import SuperfacilityAPI, SuperfacilityAccessToken
except ImportError:
    logging.error("Could not load SuperfacilityAPI")


class SFapiSlurm(Slurm):
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
        jobs = self.sfapi.get_jobs(
            user=self.user_name,
            site=self.site,
            sacct=False
        )

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

        for i in range(wait_time):
            time.sleep(1)
            task_info = self.sfapi.tasks(task_id=ret['task_id'])
            if task_info['result'] != None:
                task = json.loads(task_info['result'])
                return {"stdout": int(task['jobid']), "stderr": str(task['error']), "returncode": 0}
            logging.debug(f"Trying {i+1}")

        return {"stdout": {'task_id': int(ret['task_id'])}, "stderr": ret['error'], "returncode": 0}


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    slurm = SFapiSlurm(
        script_path='/global/homes/t/tylern/htcondor_workflow_scron'
    )

    job = slurm.sbatch(compute_type='large')
    print(job)
    job = slurm.sbatch(compute_type='gpu')
    print(job)

    squeue = slurm.squeue()
    jobs = pd.DataFrame.from_dict(squeue)

    print("Wating 10s...")
    time.sleep(10)

    to_cancel = jobs[jobs['name'].str.contains('htcondor_worker')]

    for job_id in to_cancel['jobid']:
        print(slurm.scancel(job_id=job_id))
