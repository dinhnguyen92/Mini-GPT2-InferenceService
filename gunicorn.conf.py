# Gunicorn configuration file
import multiprocessing
import math

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"

# The recommended number of workers is (cpu * 2 + 1).
# Our Azure server instance has 4 vCPU, which translates to 9 workers.
# This is a good number since each submission from the React app
# will generate 3 text completion requests, which is divisible by 9.
workers = multiprocessing.cpu_count() * 2 + 1