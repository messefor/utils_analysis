'''logging module'''

import logging


def log_args(arg_dic, logger, level):
  for k, v in arg_dic.items():
    msg = f'(args) {k}: {v}'
    getattr(logger, level)(msg)

def log_args_inlist(arg_names, logger, vars):
  for nm in arg_names:
    set_var = 'logger.{level}(r"(args) {k}: {v}")'.format(level='debug',
                                k=nm, v=vars[nm])
    eval(set_var)

def log_saved(x, logger):
  logger.info(f'Saved: {x}')


def get_logger(loger_name, level=logging.INFO):

  # create logger
  logger = logging.getLogger(loger_name)
  logger.setLevel(level)

  # create console handler and set level to debug
  ch = logging.StreamHandler()

  # create formatter
  formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

  # add formatter to ch
  ch.setFormatter(formatter)

  # add ch to logger
  logger.addHandler(ch)

  return logger
