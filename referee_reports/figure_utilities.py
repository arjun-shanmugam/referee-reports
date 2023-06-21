import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
import referee_reports.constants
def save_figure_and_close(figure: plt.Figure,
                          filename: str,
                          bbox_inches: str = 'tight'):
    """
    Helper function to save and close a provided figure.
    :param figure: Figure to save and close.
    :param filename: Location on disk to save figure.
    :param bbox_inches: How to crop figure before saving.
    """
    figure.savefig(filename, bbox_inches=bbox_inches)
    plt.close(figure)
def plot_labeled_vline(ax: Axes,
                       x: float,
                       text: str,
                       color: str = referee_reports.constants.Colors.P10,
                       linestyle: str = '--',
                       text_y_location_normalized: float = 0.85,
                       size='small',
                       zorder: int = 10):
    """

    :param ax: An Axes instance on which to plot.
    :param x: The x-coordinate of the vertical line.
    :param text: The text with which to label the vertical line.
    :param color: The color of the vertical line and of the text.
    :param linestyle: The style of the vertical line.
    :param text_y_location_normalized: The y-coordinate of the
            text as a portion of the total length of the y-axis.
    :param size: The size of the text.
    :param zorder: Passed as Matplotlib zorder argument in ax.text() call.
    """

    # Plot vertical line.
    ax.axvline(x=x,
               c=color,
               linestyle=linestyle)

    # Create blended transform for coordinates.
    transform = transforms.blended_transform_factory(ax.transData,  # X should be in data coordinates
                                                     ax.transAxes)  # Y should be in axes coordinates (between 0 and 1)

    # Plot text
    ax.text(x=x,
            y=text_y_location_normalized,
            s=text,
            c=color,
            size=size,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', lw=1, ec='black'),
            transform=transform,
            zorder=zorder)


def plot_histogram(ax: Axes,
                   x: np.ndarray,
                   xlabel: str,
                   title: str = "",
                   ylabel: str = "Relative Frequency",
                   edgecolor: str = 'black',
                   color: str = referee_reports.constants.Colors.P1,
                   summary_statistics=None,
                   summary_statistics_linecolor: str = referee_reports.constants.Colors.P3,
                   summary_statistics_text_y_location_normalized: float = 0.15,
                   decimal_places: int = 2,
                   alpha: float = 1,
                   label: str = ""):
    """
    Plot a histogram.
    :param ax: An Axes instance on which to plot.
    :param x: A Series containing data to plot as a histogram.
    :param title: The title for the Axes instance.
    :param xlabel: The xlabel for the Axes instance.
    :param ylabel: The ylabel for the Axes instance.
    :param edgecolor: The edge color of the histogram's bars.
    :param color: The fill color of the histogram's bars.
    :param summary_statistics: The summary statistics to mark on the histogram.
    :param summary_statistics_linecolor: The color of the lines marking the
            summary statistics.
    :param summary_statistics_text_y_location_normalized: The y-coordinate
            of the text labels for summary statistics lines as a portion of
            total y-axis length.
    :param decimal_places: The number of decimal places to include in summary
            statistic labels.
    :param alpha: The opacity of the histogram's bars.
    :param label: The label for the histogram to be used in creating Matplotlib
            legends.
    """

    # Check that requested summary statistics are valid.
    if summary_statistics is None:
        summary_statistics = ['min', 'med', 'max']
    else:
        if len(summary_statistics) > 3:
            raise ValueError("No more than three summary statistics may be requested.")
        for summary_statistic in summary_statistics:
            if (summary_statistic != 'min') and (summary_statistic != 'med') and (summary_statistic != 'max'):
                raise ValueError("When requesting summary statistics, please specify \'min\', \'med\', or \'max\'.")

    # Plot histogram.
    ax.hist(x,
            color=color,
            edgecolor=edgecolor,
            weights=np.ones_like(x) / len(x),
            alpha=alpha,
            label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Calculate summary statistics.
    statistics = []
    labels = []
    if 'min' in summary_statistics:
        statistics.append(np.min(x))
        labels.append(f"min: {float(np.min(x)).round(decimal_places)}")
    if 'med' in summary_statistics:
        statistics.append(np.median(x))
        labels.append(f"med: {float(np.median(x)).round(decimal_places)}")
    if 'max' in summary_statistics:
        statistics.append(np.max(x))
        labels.append(f"max: {float(np.max(x)).round(decimal_places)}")
    for statistic, label in zip(statistics, labels):
        plot_labeled_vline(ax=ax,
                           x=statistic,
                           text=label,
                           color=summary_statistics_linecolor,
                           text_y_location_normalized=summary_statistics_text_y_location_normalized)