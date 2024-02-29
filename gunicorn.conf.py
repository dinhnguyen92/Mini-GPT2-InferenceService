# Gunicorn configuration file
import multiprocessing
import math

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"

# Gunicorn and uvicorn hoad huge amounts of memory
# so instead of using (cpu * 2 + 1) as the max number of workers,
# we'll just cpu / 3 to avoid out-of-memory errors,
# especially when we need to reserve memory to load the models.
workers = math.ceil(multiprocessing.cpu_count() / 3)