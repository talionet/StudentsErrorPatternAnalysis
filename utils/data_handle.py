import numpy as np
import pandas as pd
from pandas import DataFrame as df
import sys
from scipy.spatial import distance




def pivot(raw_data, index_col,columns_col, values_col, is_agg=False, agg_function='mean', dropna=True, fill_value=None, head=None, convert_to_numeric=False):
    #prepares a table.
    if head is not None:
        raw_data=raw_data.head(head)

    if is_agg:
        X= pd.pivot_table(raw_data,values=values_col,index=index_col, columns=columns_col, aggfunc=agg_function)
    else: #for non numeric data
        data_without_duplicates=raw_data.drop_duplicates([index_col,columns_col], keep=agg_function)
        print('%i/%i duplicated rows were removed (keep=%s)' %(len(raw_data)-len(data_without_duplicates), len(raw_data),agg_function))
        X=data_without_duplicates.pivot(index=index_col, columns=columns_col, values=values_col)

    if dropna:
        X=X.dropna(how='all')
    if fill_value is not None:
        X.fillna(fill_value)
    if convert_to_numeric:
        X =X.apply(pd.to_numeric, args=('coerce',))

    return X

def mask_data(data, type='', makeZero=np.nan, make_plus=1, make_minus=0):
    data=df(data)
    #input - data of one type
    if type=='':
        return data
    if type=='bool':
        return pd.notnull(data)
    elif type=='nan_to_zero':
        data= data.applymap(lambda x: -1 if x==0 else x) #convert 0 to -1
        return data.fillna(0.)

def define_distance_metric(distance_name='sum_joint_questions', min_intersection=10):
    def _validate_vector(u, dtype=None):
        # XXX Is order='c' really necessary?
        u = np.asarray(u, dtype=dtype, order='c').squeeze()
        # Ensure values such as u=1 and u=[1] still return 1-D arrays.
        u = np.atleast_1d(u)
        if u.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        return u

    drop_na = lambda x: pd.Series(x).dropna().index


    if distance_name == 'sum_joint_questions':
        # number of items which are not Nan for BOTH u and v.
        def distfunc(u, v):
            joint_questions=set(drop_na(u)).intersection(set(drop_na(v)))
            return len(joint_questions)

    elif distance_name == 'jaccard_intersection':
        #number of non-similar responses of all responses different from zero (for BOTH v, u)
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum())< min_intersection:
                return np.nan
            else:
                dist = (np.double(np.bitwise_and((u != v),
                                                 np.bitwise_or(u != 0, v != 0)).sum()) /
                        np.double(np.bitwise_and(u != 0, v != 0).sum()))
                return dist


    elif distance_name== 'joint_minus_count_drop0':
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum()) < min_intersection:
                return np.nan
            elif np.bitwise_and(u == -1,v == -1).sum() == 0:
                return 1
            else:
                return  -(np.double(np.bitwise_and(u == -1,v == -1).sum()))


    elif distance_name== 'joint_minus_percent_drop0':
        def distfunc(u,v):
            u = _validate_vector(u)
            v = _validate_vector(v)
            if np.double(np.bitwise_and(u != 0, v != 0).sum())< min_intersection:
                return np.nan
            elif np.bitwise_and(u == -1,v == -1).sum()==0:
                return np.nan
            else:
                return  np.double(np.bitwise_and(u ==-1, v == -1).sum()) / np.double(np.bitwise_and(u == -1, v == -1).sum())

    return distfunc

def normalize_data(data, min_value=0, by='min_max', fillna=None):
    data = df(data)
    is_series= data.shape[1]==1
    if by is None:
        ndata=data
    if by=='min_max':
        ndata = (data - data.mean()) / (data.max() - data.min())
    if by == 'standard':
        ndata = (data - data.mean()) / (data.std())


    if min_value is not None:
        ndata= ndata - ndata.min() + min_value


    if fillna is not None:
        if type(fillna) == float or type(fillna) == int:
            ndata = ndata.fillna(fillna)
        if fillna == 'max_and_std':
            ndata = ndata.fillna(ndata.max() + 2 * ndata.std())
        elif fillna == 'max':
            ndata = ndata.fillna(ndata.max())


    if is_series:
        return ndata[0]
    else:
        return ndata


def pairwise_dist(X,metric):
    X = np.asarray(X, order='c')
    s = X.shape
    if len(s) == 2:

        m, n = s
        dm = np.zeros((m * (m - 1)) // 2, dtype=np.double)
        k = 0
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                dm[k] = metric(X[i], X[j])
                k = k + 1
            if i%100==0:
                df(dm).to_csv('temp_similarity_matrix_%i.csv' %i)
    else:
        dm = df(columns=X,index=X)
        i = 0
        X_not_calculated=list(X.copy())
        for d in X:
            print(i)
            dist = [None for s in range(i)]+[metric(d, d2) for d2 in X_not_calculated]
            dm.loc[d]=dist
            X_not_calculated.remove(d)
            i+=1
            if i%100==0:
                dm.to_csv('temp_similarity_matrix_%i.csv' %i)
    return dm

def add_LO_indexes_to_meta_data(meta_data=None, lo_index=None, is_load_data=True, md_file='MD_math_processed.csv', loi_file='LOs_order_index_fraction.csv', filter_by_language=True, save_to_csv=True, save_name='MD_math_processed.csv'):
    """
    adds the Learning Objective index to questions metadata, in order to know the order of questions.
    Aslo adds 'question index' - the order in which the questions appeared in CET content player using LO index + the number of question in LO session.

    :param meta_data:
    :param lo_index:
    :param is_load_data:
    :param md_file:
    :param loi_file:
    :param filter_by_language:
    :param save_to_csv:
    :param save_name:
    :return:
    """
    if is_load_data:
        meta_data= df.from_csv(os.path.join(DATA_ROOT,md_file))
        lo_index = df.from_csv(os.path.join(DATA_ROOT,loi_file))
    # lo_index.index = [s.lower() for s in lo_index.index]
    meta_data = meta_data.loc[meta_data.nLanguage == 1]
    meta_data = meta_data.reset_index(drop=True)
    lo_index_ordered = lo_index.loc[meta_data.gLO].reset_index(drop=True)
    meta_data[lo_index_ordered.columns] = lo_index_ordered
    if 'num_of_questions_in_lo_session' in meta_data.columns:
        meta_data.num_of_questions_in_lo_session=meta_data['sQuestionPageID'].apply(lambda s: int(s[s.rfind('_') + 1:]))

    meta_data['question_index'] = meta_data.LO_general_index + 0.01 * meta_data.num_of_questions_in_lo_session

    if filter_by_language:
        meta_data = meta_data.loc[meta_data.nLanguage == 1]

    meta_data.drop_duplicates(inplace=True)
    if save_to_csv:
        meta_data.to_csv(os.path.join(DATA_ROOT,save_name))
    return meta_data