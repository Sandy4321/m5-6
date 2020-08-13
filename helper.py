import numpy as np

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def build_df(main_df, calendar_df, price_df, store_id_):
    out_df = main_df.copy()
    out_df = out_df.loc[lambda out_df: out_df.store_id == store_id_]
    out_df = out_df.melt(['id', 'item_id', 'dept_id', 'cat_id', 'store_id','state_id'],
                         var_name='d', value_name='demand')
    out_df = out_df.merge(right=calendar_df, how='outer', on='d')
    out_df = out_df.merge(right=price_df, how='outer', on=['store_id', 'item_id', 'wm_yr_wk'])
    out_df.d = out_df.d.astype(str)
    return reduce_mem_usage(out_df)

def train_test_split(input_df, train_size, test_size):
    input_df.d = input_df.d.astype(str)
    train_df = input_df[input_df.d.isin(["d_" + str(a) for a in np.arange(1, train_size+1)])]
    test_df = input_df[input_df.d.isin(["d_" + str(a) for a in np.arange(train_size+1, train_size+test_size+2)])]
    return train_df, test_df
