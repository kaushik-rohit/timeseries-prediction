import dm_test
import pandas as pd
# ,   , 
symbols = [ 'ACC', 'ULTRACEMCO', 'CIPLA', 'HDFC', 'HCLTECH', 'JSWSTEEL', 'MARUTI','INFY', 'BHARTIARTL', 'AXISBANK']

def test():
    rows = []
    for sym in symbols:
        row = [sym]
        df = pd.read_csv('./prediction/pred_{}.csv'.format(sym))
        actual = df['actual'].tolist()
        ann= df['ann'].tolist()
        cnn = df['cnn'].tolist()
        lstm = df['lstm'].tolist()
        gru = df['gru'].tolist()
        
        param1 = dm_test.dm_test(actual, ann, cnn)
        param2 = dm_test.dm_test(actual, lstm, gru)
        param3 = dm_test.dm_test(actual, ann, lstm)
        
        row += [param1, param2, param3]
        rows.append(row)
        
    res = pd.DataFrame(rows, columns=['stock', 'ann-cnn', 'lstm-gru', 'ann-lstm'])
    res.to_csv(path_or_buf='./hypothesis_test_results.csv')
if __name__ == '__main__':
    test()
