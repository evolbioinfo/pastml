import logging


def set_up_pastml_logger(verbose, default_level=logging.ERROR):
    logger = logging.getLogger('pastml')
    logger.setLevel(level=logging.DEBUG if verbose else default_level)
    logger.propagate = False
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(asctime)s %(message)s', datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
