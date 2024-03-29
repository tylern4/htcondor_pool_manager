# from jaws_condor import config
from typing import Dict, List
from cmd_utils import run_command
from loguru import logger
import pandas as pd
import time


class HTCondorError(Exception):
    pass


class HTCondorCmdFailed(HTCondorError):
    pass


class HTCondor:
    def __init__(self, columns, **kwargs):
        self.columns = columns

    def condor_q(self):
        # Call the command
        condor_q_cmd = f"condor_q -allusers -af {self.columns}"
        try:
            condor_jobs = self.call_condor_command(condor_q_cmd)
            # get serialized and processed dataframe
            dataframe = process_condor_q(condor_jobs, self.condor_columns())
        except HTCondorCmdFailed as err:
            logger.error(
                f"{type(err).__name__} make sure condor is running and CONDOR_CONFIG is set properly"
            )
            dataframe = process_condor_q([], self.condor_columns())

        return dataframe

    def condor_idle(self):
        get_idle = 'condor_status -const "TotalSlots == 1" -af Machine'
        try:
            idle_list = self.call_condor_command(get_idle)
            # Takes a list of lists to a single list
            idle = [item for sublist in idle_list for item in sublist]
        except HTCondorCmdFailed as err:
            logger.error(
                f"{type(err).__name__} make sure condor is running and CONDOR_CONFIG is set properly"
            )
            idle = []
        return idle

    def condor_columns(self):
        # Splits the string columns to be used as header of dataframe
        return self.columns.split()

    def call_condor_command(self, condor_cmd: str) -> List:
        # Call the condor_q with the desired options
        stdout, stderr, returncode = run_command(condor_cmd)
        # The usual failures
        if returncode != 0:
            logger.critical(
                f"ERROR: failed to execute condor_q command: {condor_cmd}")
            raise HTCondorCmdFailed(
                f"{condor_cmd} command failed with {stderr}")

        return parse_condor_outputs(stdout)


def parse_condor_outputs(stdout: str) -> List:
    # split outputs by rows
    outputs = stdout.split("\n")
    # Split each row into columns
    condor_jobs = [job.split() for job in outputs]
    # Removes columns with no values (Usually the last column)
    condor_jobs = [q for q in condor_jobs if len(q) != 0]
    return condor_jobs


def process_condor_q(condor_jobs: List, columns: List) -> List[Dict]:
    # if there are no jobs send back empty array of jobs
    if len(condor_jobs) == 0:
        return [{k: 0 for k in columns}]

    # Create a dataframe from the split outputs
    df = pd.DataFrame(condor_jobs, columns=columns)

    # Change the types
    df["JobStatus"] = df["JobStatus"].astype(int)
    df["RequestMemory"] = df["RequestMemory"].astype(float) / 1024
    df["MemoryUsage"] = df["MemoryUsage"].str.replace("undefined", "0.0")
    df["MemoryUsage"] = df["MemoryUsage"].astype(float) / 1024
    df["RequestCpus"] = df["RequestCpus"].astype(float)
    df["RemoteSysCpu"] = df["RemoteSysCpu"].astype(float)
    df["RemoteUserCpu"] = df["RemoteUserCpu"].astype(float)

    now = int(time.time())
    df["JobStartDate"] = df["JobStartDate"].str.replace("undefined", str(now))
    df["total_running_time"] = now - df["JobStartDate"].astype(int)
    df["cpu_percentage"] = (((df["RemoteSysCpu"] + df["RemoteUserCpu"]) /
                            df["RequestCpus"])
                            / df["total_running_time"]) * 100
    df["total_q_time"] = df["JobStartDate"].astype(
        int) - df["QDate"].astype(int)

    # Returns a serilized/json version better for passing between objects
    # And for testing
    return df.to_dict("records")
