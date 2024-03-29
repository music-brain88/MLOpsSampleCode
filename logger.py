from PIL import Image
import numpy as np
import pandas as pd
import logging
import os

from trains import Task
task = Task.init(project_name='MLOps', task_name='logging sample')

logger = Task.current_task().get_logger()
# report scalar values
logger.report_scalar("example_scalar", "series A", iteration=0, value=100)
logger.report_scalar("example_scalar", "series A", iteration=1, value=200)

# report histogram
histogram = np.random.randint(10, size=10)
logger.report_histogram("example_histogram", "random histogram", iteration=1, values=histogram,
                        xaxis="title x", yaxis="title y")

# report confusion matrix
confusion = np.random.randint(10, size=(10, 10))
logger.report_matrix("example_confusion", "ignored", iteration=1, matrix=confusion, xaxis="title X", yaxis="title Y")

# report 3d surface
logger.report_surface("example_surface", "series1", iteration=1, matrix=confusion,
                      xaxis="title X", yaxis="title Y", zaxis="title Z")

# report 2d scatter plot
scatter2d = np.hstack((np.atleast_2d(np.arange(0, 10)).T, np.random.randint(10, size=(10, 1))))
logger.report_scatter2d("example_scatter", "series_xy", iteration=1, scatter=scatter2d,
                        xaxis="title x", yaxis="title y")

# report 3d scatter plot
scatter3d = np.random.randint(10, size=(10, 3))
logger.report_scatter3d("example_scatter_3d", "series_xyz", iteration=1, scatter=scatter3d,
                        xaxis="title x", yaxis="title y", zaxis="title z")

logger.flush()
