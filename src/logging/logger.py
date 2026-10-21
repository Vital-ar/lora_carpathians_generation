import logging 
from datetime import datetime
import os
LOG_FILE = f'{datetime.now().strftime('%d_%m_%Y')}.log'
log_path = os.path.join('logs')
os.makedirs(log_path, exist_ok= True)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] ::: %(levelname)s ::: FILE [%(name)s] --- LINE: [%(lineno)d]\n%(message)s",
    level = logging.INFO
)