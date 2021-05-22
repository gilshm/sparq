import numpy as np
import warnings


class StatsLogger:
    def __init__(self):
        self._tbls = {}

    def add_tbl(self, tbl_name: str):
        if tbl_name in self._tbls:
            return False
        else:
            self._tbls[tbl_name] = {'headers': None, 'rows': []}
            return True

    def add_row(self, tbl_name: str, row: list):
        if isinstance(row, np.ndarray):
            row = row.tolist()

        if not isinstance(row, list):
            warnings.warn('Row was not inserted, must be a list', RuntimeWarning)
            return False
        else:
            self._tbls[tbl_name]['rows'].append(row)

    def add_headers(self, tbl_name: str, headers: list):
        if isinstance(headers, np.ndarray):
            headers = headers.tolist()

        if not isinstance(headers, list):
            warnings.warn('Headers were not set, must be a list', RuntimeWarning)
            return False
        else:
            self._tbls[tbl_name]['headers'] = headers

    def get_tbl(self, tbl_name: str):
        return self._tbls[tbl_name].copy()

    def get_tbl_normalized(self, tbl_name: str, norm_by_col: list, precision=2):
        tbl = self.get_tbl(tbl_name)
        new_tbl = {'headers': None, 'rows': []}

        new_headers = []
        for col in tbl['headers']:
            new_headers.append(col)
            new_headers.append('%')
        new_tbl['headers'] = new_headers

        for row in tbl['rows']:
            denominator = 0
            for col in norm_by_col:
                idx = tbl['headers'].index(col)
                denominator += row[idx]

            if denominator == 0:
                denominator = float("inf")

            new_row = []
            for col in row:
                new_row.append(col)
                new_row.append(np.around(100 * col / denominator, precision))

            new_tbl['rows'].append(new_row)

        return new_tbl

    def get_tbl_sum(self, tbl_name: str, show_cols=None):
        tbl_sum = np.array(self._tbls[tbl_name]['rows']).sum(axis=0)
        tbl = {'headers': self._tbls[tbl_name]['headers'], 'rows': [tbl_sum]}

        if show_cols is not None:
            new_tbl = {'headers': [], 'rows': [[]]}
            for col in show_cols:
                idx = tbl['headers'].index(col)
                new_tbl['headers'].append(col)
                new_tbl['rows'][0].append(tbl['rows'][0][idx])

            tbl = new_tbl

        return tbl.copy()

    def get_tbl_sum_normalized(self, tbl_name: str, norm_by_col: list, precision=2):
        tbl = self.get_tbl_sum(tbl_name)

        denominator = 0
        for col in norm_by_col:
            args = col.split(':')
            idx = self._tbls[args[0]]['headers'].index(args[1])

            tbl_other = self.get_tbl_sum(args[0])
            denominator += tbl_other['rows'][0][idx]

        tbl['rows'] = [np.around(100 * tbl['rows'][0] / denominator, precision)]

        return tbl

    def clear_all(self):
        for k in self._tbls.keys():
            self.clear_tbl(k)

    def clear_tbl(self, tbl_name: str):
        self._tbls[tbl_name] = {'headers': None, 'rows': []}

    def clear_tbl_rows(self, tbl_name: str):
        self._tbls[tbl_name]['rows'] = []

    def clear_tbl_headers(self, tbl_name: str):
        self._tbls[tbl_name]['headers'] = None

    def del_tbl(self, tbl_name: str):
        del self._tbls[tbl_name]
