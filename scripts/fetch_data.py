from nsepy import get_history
from datetime import date
import time

symbols = ['ULTRACEMCO', 'CIPLA', 'ACC', 'HDFC', 'HCLTECH', 'JSWSTEEL', 'MARUTI', 'INFY', 'BHARTIARTL', 'AXISBANK']


def get_data():
    for sym in symbols:
        print('fetching data for {}'.format(sym))
        try:
            data = get_history(symbol=sym, start=date(2002, 1, 1), end=date(2019, 1, 15))
        except:
            print('----------------Sleeping for 5 minutes. Too much Load-------------------------')
            time.sleep(300)
            data = get_history(symbol=sym, start=date(2002, 1, 1), end=date(2019, 1, 15))
        print('data fetched!!!')
        data.to_csv(path_or_buf='../data/{}.csv'.format(sym))


def main():
    get_data()


if __name__ == '__main__':
    main()
