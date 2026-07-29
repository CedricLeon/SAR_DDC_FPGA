# EPSILON used in log computations to avoid log(0)
EPS = 1e-2

# ----- Normalization constants computed on whole datasets -----
# MERLIN constants: global maximum and minimum values obtained empirically and used for the training of MERLIN (see https://github.com/hi-paris/deepdespeckling)
m = -1.429329123112601
M = 10.089038980848645

# Amplitude: log(sqrt(intensity) + 1e-2)
AMP_MIN = 4.605170249938965
AMP_MAX = 10.742239952087402
AMP_LIN_99 = 545.2018433569272

# ################ /!\ DEPRECATED /!\ ################
# # ----- Intensity: log(a^2 + b^2 + epsilon)-----
# # These percentiles are the statistics of the intensity image in log-scale of the whole dataset. To know how they were obtained, refer to the README.
# # To avoid error in log we need to add a small epsilon. The value of this epsilon significantly impacts the minimum value of the log image.
# inten_min_1e2 = -4.605170249938965      # log + 1e-2
# inten_min_1e3 = -6.907755374908447      # log + 1e-3
# inten_min_1e6 = -13.81551074981689      # log + 1e-6
# inten_min_spacing = -36.04365338911715  # log + np.spacing(1)
# inten_max_1e2 = 21.484479904174805      # max is barely impacted by the value of epsilon

# # ----- Percentiles for log(intensity + np.spacing(1)), used in old scripts. -----
# PERCENTILES = {
#     "p1": 5.0998743307292065,
#     "p5": 6.76272625759834,
#     "p10": 7.5093313731365825,
#     "p25": 8.58166854680157,
#     "p50": 9.592196081062,
#     "p75": 10.47517273375431,
#     "p90": 11.218084495319967,
#     "p95": 11.667549457990336,
#     "p99": 12.611018051940837,
# }
