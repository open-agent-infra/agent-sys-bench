from ..persistence_interface import IMonitorPersistence, MonitorConfig
from ...record import E2EDelayRecord, RemoteDelayRecord

import csv
import os
import numpy as np

DEFAULT_PATH = "./monitor"
E2E_DELAY_FILE = "e2e_delay.csv"
REMOTE_DELAY_FILE = "remote_delay.csv"

E2E_DELAY_HEADER = [
    "timestamp",
    "request_id",
    "parent_service_name",
    "service_name",
    "delay",
]
REMOTE_DELAY_HEADER = [
    "timestamp",
    "request_id",
    "parent_service_name",
    "service_name",
    "delay",
]


def auto_analysis(store_dir: str):
    # store stats by service name
    stats_dir = store_dir + os.path.sep + "stats"
    os.makedirs(stats_dir, exist_ok=True)

    def generate_stats(
        filename: str,
    ):
        colnames = (
            E2E_DELAY_HEADER if filename == E2E_DELAY_FILE else REMOTE_DELAY_HEADER
        )
        file_path = store_dir + os.path.sep + filename

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # convert the reader to a list of dictionaries
            data = [dict(zip(colnames, row)) for row in reader]
            group = {}
            for d in data:
                if d["service_name"] not in group:
                    group[d["service_name"]] = []
                group[d["service_name"]].append(
                    {
                        **d,
                        "delay": float(d["delay"]),
                        "end_ts": float(d["timestamp"]),
                        "start_ts": float(d["timestamp"]) - float(d["delay"]),
                    }
                )
            percentiles = [25, 50, 75, 90, 95, 99]
            stats = []
            for name, group in group.items():
                group_st = min([d["start_ts"] for d in group])
                group_et = max([d["end_ts"] for d in group])
                total_records = len(group)
                total_time = group_et - group_st
                throughput = total_records / total_time
                delay_stats = np.percentile([d["delay"] for d in group], percentiles)
                stats.append(
                    {
                        "service_name": name,
                        "throughput": throughput,
                        "avg_delay": np.mean([d["delay"] for d in group]),
                        "std_delay": np.std([d["delay"] for d in group]),
                        "max_delay": max([d["delay"] for d in group]),
                        "min_delay": min([d["delay"] for d in group]),
                        "25%": delay_stats[0],
                        "50%": delay_stats[1],
                        "75%": delay_stats[2],
                        "90%": delay_stats[3],
                        "95%": delay_stats[4],
                        "99%": delay_stats[5],
                        "start_ts": group_st,
                        "end_ts": group_et,
                        "total_time": total_time,
                        "total_records": total_records,
                    }
                )
            if len(stats) > 0:
                stats = sorted(stats, key=lambda x: x["start_ts"])
                with open(
                    stats_dir + os.path.sep + filename,
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=stats[0].keys())
                    writer.writeheader()
                    writer.writerows(stats)

    generate_stats(filename=E2E_DELAY_FILE)
    generate_stats(filename=REMOTE_DELAY_FILE)


class CSVMonitorPersistence(IMonitorPersistence):

    e2e_delay_file = None
    remote_delay_file = None

    e2e_delay_writer = None
    remote_delay_writer = None

    def store_e2e_delay(self, record: E2EDelayRecord):
        self.e2e_delay_writer.writerow(record.to_tuple())

    def store_remote_delay(self, record: RemoteDelayRecord):
        self.remote_delay_writer.writerow(record.to_tuple())

    def execute(self, config: MonitorConfig, *args, **kwargs):
        store_dir = DEFAULT_PATH
        if config is not None and config.store_dir is not None:
            store_dir = config.store_dir
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        self.e2e_delay_file = open(
            store_dir + os.path.sep + E2E_DELAY_FILE,
            "a",
            encoding="utf-8",
            newline="",
        )
        self.remote_delay_file = open(
            store_dir + os.path.sep + REMOTE_DELAY_FILE,
            "a",
            encoding="utf-8",
            newline="",
        )

        self.e2e_delay_writer = csv.writer(self.e2e_delay_file)
        self.remote_delay_writer = csv.writer(self.remote_delay_file)

        self.store_dir = store_dir

    def close(self):
        self.e2e_delay_file.close()
        self.remote_delay_file.close()

        auto_analysis(store_dir=self.store_dir)

    def flush(self):
        self.e2e_delay_file.flush()
        os.fsync(self.e2e_delay_file.fileno())
        self.remote_delay_file.flush()
        os.fsync(self.remote_delay_file.fileno())
