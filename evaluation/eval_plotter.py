import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.lcer import get_output_images, get_base_path


def get_correct_path(file_name):
    return os.path.join(get_base_path(), get_output_images(), "{}.pdf".format(file_name))


def start_barplot(data: pd.DataFrame, save_images, horizontal=False, prefix=""):
    # Preprocess
    data = data.sort_values(by=[data.columns[-1]])  # sort by last column (last column should be error)

    # Plot
    fig, ax = plt.subplots()
    if horizontal:
        data.plot.barh(x=data.columns[0], y=data.columns[-1], ax=ax)
    else:
        data.plot.bar(x=data.columns[0], y=data.columns[-1], ax=ax)

    # Plot Style
    ax.set_title("PPD-RSME Results for different Models")
    ax.get_legend().remove()
    if horizontal:
        plt.ylabel("Model Name")
        plt.xlabel("PPD-RSME")
    else:
        plt.xticks(rotation=70)
        plt.xlabel("Model Name")
        plt.ylabel("PPD-RSME")

    fig.tight_layout()

    if save_images:
        name = prefix
        name += "error_model_barplot"
        if horizontal:
            name += "_horizontal"
        plt.savefig(get_correct_path(name))

    # Show
    plt.show()


def tableplot(data: pd.DataFrame, save_images, change_perspective, prefix=""):
    # Idea from: https://stackoverflow.com/a/45936469

    # Add Change column
    col_rsme = data.columns[-1]
    value_cpov = data[data[data.columns[0]] == change_perspective][col_rsme].values[0]
    data["Reduction in PPD-RSME"] = (value_cpov - data[col_rsme]) / value_cpov * 100
    data = data.sort_values(by=["Reduction in PPD-RSME"])
    data["Reduction in PPD-RSME"] = data["Reduction in PPD-RSME"].apply(lambda x: '{:.2f}%'.format(x))
    data[col_rsme] = data[col_rsme].apply(lambda x: '{:.3f}'.format(x))

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=data.values, colLabels=data.columns, loc='center')

    if save_images:
        name = prefix
        name += "default_table_plot"
        plt.savefig(get_correct_path(name))

    fig.tight_layout()
    plt.show()

    return data


def aggregated_barplot(data: pd.DataFrame, save_images, agg_by, prefix=""):
    # Preprocess
    # Aggregate

    data = data.groupby(agg_by, as_index=False).mean()

    data = data.sort_values(by=[data.columns[-1]])  # sort by last column (last column should be error)

    # Plot
    fig, ax = plt.subplots()
    data.plot.barh(x=data.columns[0], y=data.columns[-1], ax=ax)

    # Plot Style
    ax.set_title("Mean PPD-RSME Results for different categories")
    ax.get_legend().remove()
    plt.ylabel("Model Name")
    plt.xlabel("Mean PPD-RSME")
    fig.tight_layout()

    if save_images:
        name = prefix
        name += "agg_error_model_barplot"
        plt.savefig(get_correct_path(name))

    # Show
    plt.show()