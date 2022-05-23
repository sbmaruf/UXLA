import time
import json
import subprocess

prev_time = time()
while True:
    curr_time = time()
    diff = int(curr_time-prev_time)
    if diff == 300:
        json_data = subprocess.check_output("gpustat --json", shell=True)