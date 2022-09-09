from subprocess import Popen, PIPE


def run_command(cmd):
    std_out = None
    std_err = None
    exit_code = 1

    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)

    std_out, std_err = p.communicate()
    if isinstance(std_out, bytes):
        std_out = std_out.decode()
    if isinstance(std_err, bytes):
        std_err = std_err.decode()
    exit_code = p.returncode

    return std_out, std_err, exit_code


if __name__ == '__main__':
    print(run_command("ls -la"))
