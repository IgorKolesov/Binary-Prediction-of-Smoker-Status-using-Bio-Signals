import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hist(column: pd.DataFrame,
              bins_value: int = 20,
              xl: str = 'Value',
              yl: str = 'Frequency',
              is_grid: bool = True,
              is_subplot: bool = False,
              subplot_indexes: list = None) -> None:

    if is_subplot:
        if subplot_indexes is None:
            subplot_indexes = [1, 1, 1]
        plt.subplot(*subplot_indexes)

    plt.grid(is_grid, alpha=0.3, zorder=0)

    plt.hist(column, bins=bins_value)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f'Histogram of feature {column.name}')


def plot_boxplot(column: pd.DataFrame,
                 yl: str = 'Value',
                 is_grid: bool = True,
                 is_subplot: bool = False,
                 subplot_indexes: list = None) -> None:

    if is_subplot:
        if subplot_indexes is None:
            subplot_indexes = [1, 1, 1]
        plt.subplot(*subplot_indexes)

    plt.grid(is_grid, alpha=0.3, zorder=0)

    plt.boxplot(column)
    plt.xlabel(column.name)
    plt.ylabel(yl)
    plt.title(f'Boxplot of feature {column.name}')


def plot_kde(column: pd.DataFrame,
             xl: str = 'Value',
             yl: str = 'Density',
             is_grid: bool = True,
             is_fill: bool = True,
             is_subplot: bool = False,
             subplot_indexes: list = None) -> None:

    if is_subplot:
        if subplot_indexes is None:
            subplot_indexes = [1, 1, 1]
        plt.subplot(*subplot_indexes)

    plt.grid(is_grid, alpha=0.3, zorder=0)

    sns.kdeplot(column, fill=is_fill)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(f'Density plot of {column.name}')


def add_info_to_plot(text: str = '',
                     is_subplot: bool = False,
                     subplot_indexes: list = None) -> None:

    if is_subplot:
        if subplot_indexes is None:
            subplot_indexes = [1, 1, 1]
        plt.subplot(*subplot_indexes)

    plt.axis('off')

    plt.text(0.5, 0.5, text, ha='center', va='center')


if __name__ == '__main__':
    # read data from file
    data = pd.read_csv('../dataset/train.csv')

    columns_list = list(data.columns)  # list of columns names
    columns_list.pop(0)  # remove index col
    columns_list.pop(-1)  # remove target col
    print(columns_list)

    # Some information about data
    # We can see that there are no Nullable values
    # All columns have correct type
    print(data.info())

    # Let's take a look at the distribution plot and boxplot of the features.
    for col in columns_list:
        plt.subplots(2, 2)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        plot_hist(data[col], is_subplot=True, subplot_indexes=[2, 2, 1])
        plot_boxplot(data[col], is_subplot=True, subplot_indexes=[2, 2, 2])
        plot_kde(data[col], is_subplot=True, subplot_indexes=[2, 2, 3])
        add_info_to_plot(data[col].describe(), is_subplot=True, subplot_indexes=[2, 2, 4])

        plt.savefig(f'./graphics/{col}.png')
        plt.close()

