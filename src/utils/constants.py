# These percentiles are the statistics of the intensity image in log-scale of the whole dataset. See scripts/compute_dataset_stats.py and data/analysis/dataset_all_stats.log
PERCENTILES = {
    "p1": 5.0998743307292065,
    "p5": 6.76272625759834,
    "p10": 7.5093313731365825,
    "p25": 8.58166854680157,
    "p50": 9.592196081062,
    "p75": 10.47517273375431,
    "p90": 11.218084495319967,
    "p95": 11.667549457990336,
    "p99": 12.611018051940837,
}

# Global maximum and minimum values obtained empirically and used to normalize SAR images
M = 10.089038980848645
m = -1.429329123112601
