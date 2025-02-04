import zipfile
import requests
import io
import os
import datetime
import pandas as pd

# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
                                    axis=1)
    return raw_data

def _store_data(df):
    df.to_csv('data/raw/bike_sharing.csv', index=False)


if __name__ == '__main__':
    raw_data = _process_data(_fetch_data())
    _store_data(raw_data)
