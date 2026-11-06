"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros_like(vectors[0], dtype=np.result_type(matrices, vectors))
    for matrix, vector in zip(matrices, vectors):
        result += np.asarray(matrix) @ np.asarray(vector)
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    matrix = np.random.default_rng(seed).normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def plot_matrix_histograms(data: MatrixStatistics) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(data.matrix)
    figure, axes = plt.subplots(1, 2)
    for row in matrix:
        axes[0].hist(row, alpha=0.5)
    for column in matrix.T:
        axes[1].hist(column, alpha=0.5)
    axes[0].set_title("Rows")
    axes[1].set_title("Columns")
    figure.tight_layout()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=int)
    image[:] = background_color
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=int)
    image[:] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    center_x, center_y = (image_width - 1) / 2, (image_height - 1) / 2
    inside = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values)
    maxima = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1
    minima = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    columns = int(labels.max()) + 1 if class_count is None and labels.size else (class_count or 0)
    result = np.zeros((labels.size, columns), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1
    return result
