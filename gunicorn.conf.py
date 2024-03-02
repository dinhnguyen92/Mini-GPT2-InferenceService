# Gunicorn configuration file

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"

# Gunicorn and uvicorn hoad huge amounts of memory
# so instead of using the recommended (cpu * 2 + 1) as the max number of workers,
# we'll just 3 (to generate 3 completions in parallel) to avoid out-of-memory errors,
# especially when we need to reserve memory to load the models.
workers = 3