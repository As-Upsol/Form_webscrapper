bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = "info"
accesslog = "-"  # log to stdout
errorlog = "-"   # log to stderr
# timeout = 60  # Uncomment and adjust if you expect long requests 