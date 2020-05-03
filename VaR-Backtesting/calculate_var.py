import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path',
                    type=str,
                    required=True,
                    help='the path where stocks csv data is located')

parser.add_argument('-cf', '--confidence-level',
                    type=float,
                    required=True,
                    help='value at risk confidence level')


def calculate_value_at_risk(data, confidence_interval):
    data['returns'] = data.Close.pct_change()
    mean = np.mean(data['returns'])
    std_dev = np.std(data['returns'])

    data['returns'].hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5)
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    plt.plot(x, norm.pdf(x, mean, std_dev), "r")
    plt.show()

    var = norm.ppf(1-confidence_interval, mean, std_dev)
    print(var)


def main():
    args = parser.parse_args()

    path = args.data_path
    confidence_level = args.confidence_level
    stocks_data = pd.read_csv(path)
    stocks_data = stocks_data[['Date', 'Symbol', 'Close']]
    calculate_value_at_risk(stocks_data, confidence_level)


if __name__ == "__main__":
    main()
