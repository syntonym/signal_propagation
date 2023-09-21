from dataclasses import dataclass
import polars as pl
import os.path

DEFAULT_REASON = 'No Directionality'

@dataclass
class Record:
    category: str
    organoid: str
    threshold: float
    useable: bool
    checked: bool
    reason: str
    peaks: int
    average: int
    check_peak_detection: bool
    speed_estimate_weighted: float
    speed_estimate_unweighted: float
    new: bool

class Database:

    def __init__(self, path):
        self._path = path
        self._columns = [('category', pl.Utf8),
                         ('organoid', pl.Utf8),
                         ('threshold', pl.Float64),
                         ('useable', pl.Boolean),
                         ('checked', pl.Boolean),
                         ('reason', pl.Utf8),
                         ('peaks', pl.Int64),
                         ('average', pl.Int64),
                         ('check_peak_detection', pl.Boolean),
                         ("speed_estimate_weighted", pl.Float64),
                         ("speed_estimate_unweighted", pl.Float64)]

        self._column_names = [name for name, dtype in self._columns]
        self._read()

    def _read(self):
        if self._path.endswith('.csv'):
            self._format = 'csv'
        elif self._path.endswith('.parquet'):
            self._format = 'parquet'
        else:
            raise Exception('Cannot detect database format')

        if os.path.exists(self._path):
            if self._format == 'csv':
                self.df = pl.read_csv(self._path)
            elif self._format == 'parquet':
                self.df = pl.read_parquet(self._path)
        else:
            self.df = pl.DataFrame(columns=self._columns)

        for column_name, dtype in self._columns:
            if column_name not in self.df.columns:
                print(column_name)
            assert column_name in self.df.columns

    def get_reasons(self):
        reasons = list(self.df['reason'].unique())
        if None in reasons:
            reasons.remove(None)
        reasons = sorted(reasons)
        if DEFAULT_REASON not in reasons:
            reasons.append(DEFAULT_REASON)
        reasons.append('new')
        return reasons

    def get(self, organoid):
        s = self.df.filter(pl.col('organoid') == organoid)
        if len(s) == 0:
            record = Record('', '', 0.25, True, False, DEFAULT_REASON, 1, 0, False, True, -2)
        elif len(s) == 1:
            record = Record(*[s[name][0] for name, dtype in self._columns], False)
        else:
            raise Exception(f'Data Corruption: Found {len(s)} many organoids for {organoid}')
        return record

    def save(self, record):
        tmp_df = self.df.filter(pl.col('organoid') != record.organoid)
        new_row = pl.DataFrame({key: [getattr(record,key)] for key, dtype in self._columns}, columns=self._columns)
        tmp_df = pl.concat([tmp_df, new_row])
        self.df = tmp_df
        self.persist()

    def persist(self):
        if self._format == 'csv':
            self.df.write_csv(self._path)
        elif self._format == 'parquet':
            self.df.write_parquet(self._path)
        else:
            raise Exception(f'Unknown format: {self._format}')

if __name__ == '__main__':
    Database('database.parquet')
