import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.lcer import get_output_images


def start_barplot(data: pd.DataFrame, save_images, horizontal=False):
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
        name = "error_model_barplot"
        if horizontal:
            name += "_horizontal"
        plt.savefig(os.path.join(get_output_images(), "{}.pdf".format(name)))

    # Show
    plt.show()


def tableplot(data: pd.DataFrame, save_images, change_perspective):
    # Idea from: https://stackoverflow.com/a/45936469

    # Add Change column
    col_rsme = data.columns[-1]
    value_cpov = data[data[data.columns[0]] == change_perspective][col_rsme].values[0]
    data["Reduction in PPD-RSME"] = (value_cpov - data[col_rsme])/value_cpov*100
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
        name = "default_table_plot"
        plt.savefig(os.path.join(get_output_images(), "{}.pdf".format(name)))

    fig.tight_layout()
    plt.show()

    return data
