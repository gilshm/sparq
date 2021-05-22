import Config as cfg
import datetime
import sys
import os


class Logger:
    def __init__(self):
        self.path = None
        self.log = None
        self.terminal = sys.stdout

    def write(self, msg, date=True, terminal=True, log_file=True):
        if date:
            curr_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            msg = '[{}] {}'.format(curr_time, msg)

        msg = msg + '\n'

        if terminal:
            self.terminal.write(msg)
            self.terminal.flush()

        if log_file and self.log is not None:
            self.log.write(msg)

    def write_title(self, msg, terminal=True, log_file=True, pad_width=40, pad_symbol='-'):
        self.write('', date=False)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write(' {} '.format(msg).center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write('', date=False)

    def start_new_log(self, path=None, name=None, no_logfile=False):
        self._create_log_dir(path, name)

        if no_logfile:
            self.close_log()
        else:
            self._update_log_file()

        self.write(cfg.USER_CMD)
        self.write('', date=False)

    def close_log(self):
        if self.log is not None:
            self.log.close()
            self.log = None

        return self.path

    def _update_log_file(self):
        self.close_log()
        self.log = open("{}/logfile.log".format(self.path), "a")

    def _create_log_dir(self, path=None, name=None):
        if path is None:
            dir_name = ''
            if name is not None:
                dir_name = dir_name + name + '_'
            dir_name = dir_name + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            self.path = '{}/{}'.format(cfg.RESULTS_DIR, dir_name)
        else:
            self.path = path

        os.makedirs('{}'.format(self.path))
        self.write("New results directory created @ {}".format(self.path))

