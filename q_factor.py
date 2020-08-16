import pandas as pd
import numpy as np

monthly = pd.read_csv('./monthly.csv', encoding='cp949', index_col=0, parse_dates=True)
quarterly = pd.read_csv('./quarterly.csv', encoding='cp949', index_col=0, parse_dates=True)

monthly_rtn = pd.read_csv('./monthly_rtn.csv',index_col=0, encoding='cp949', parse_dates=True)
market_cap = pd.read_csv('./market_cap.csv',index_col=0, encoding='cp949', parse_dates=True)
close = monthly[monthly.iloc[:,0] == '종가'].drop(monthly.columns[0], axis=1).T.astype('float')
close.index = pd.to_datetime(close.index)
penny_mask = close.applymap(lambda x: False if np.isnan(x) or x < 1000 else True).loc['2000':'2019']

five_percent = market_cap.apply(lambda x: np.nanpercentile(x, 5), axis=1)
microcap_mask = market_cap.gt(five_percent, axis=0)

market_cap = market_cap[microcap_mask & penny_mask]


q_index = pd.DatetimeIndex([x for x in monthly_rtn.loc['2001-01':].index if x.month == 3 or x.month == 6 or x.month == 9 or x.month == 12])
market_cap_q = market_cap.loc[q_index]


kospi = monthly[monthly.iloc[:,0]=='시장구분'].drop(monthly.columns[0], axis=1).T.replace('KS',True).replace(['EX','KQ'],[False, False]).astype('bool')
kospi.index = pd.to_datetime(kospi.index)
kospi = kospi.resample('Q').apply(lambda x: x[-1]).loc['2001':'2019']
kospi.index = q_index

mkt_rf = pd.read_csv('./kospi_rf.csv', encoding='cp949',index_col=0)

mkt_price = mkt_rf['코스피']
mkt_rtn = mkt_price.diff(1)/mkt_price.shift(1).replace(0, np.nan)
rf_m = (1+mkt_rf['시장금리:통화안정(364일)(%)']/100)**(1/12)-1

mkt_factor = mkt_rtn-rf_m
mkt_factor.index = pd.to_datetime(mkt_factor.index)
mkt_factor = mkt_factor.loc['2000':'2019']

y_index = [x for x in q_index if x.month == 6]
me_y = market_cap_q.loc[y_index]

kospi_y = kospi.loc[y_index]
me_kospi_medi = me_y[kospi_y].median(axis=1)

size = me_y.copy()
size[:] = np.nan
size = size.astype('str')
size[me_y.gt(me_kospi_medi, axis=0)] = 'B'
size[me_y.le(me_kospi_medi, axis=0)] = 'S'
size = size.replace('nan', np.nan)

roe = pd.read_csv('./anomalies/roe.csv', encoding='cp949',index_col=0, parse_dates=True).loc[:'2019']
roe_m = roe.resample('M').fillna(method='ffill',limit=3)
roe_m.index = monthly_rtn.loc['2000-06':].index
roe_m = roe_m.dropna(how='all')

total_asset = quarterly[quarterly.iloc[:, 0] == '자산총계'].drop(quarterly.columns[0], axis=1).T.astype('float')

iva = total_asset.diff(4)/total_asset.shift(4)
iva = iva.iloc[4:-2]
iva.index = q_index
iva_y = iva.shift(1).loc[y_index]


def divide_percentile(size, fac, kospi, lb, ub, name_list):
    result = size.astype('object').copy()
    crit_lb = fac[kospi].apply(lambda x: np.nanpercentile(x, lb), axis=1)
    crit_ub = fac[kospi].apply(lambda x: np.nanpercentile(x, ub), axis=1)

    lb = fac.lt(crit_lb,axis=0)
    neu = fac.ge(crit_lb, axis=0) & fac.lt(crit_ub, axis=0)
    ub = fac.ge(crit_ub, axis=0)

    result[lb] = result[lb].applymap(lambda x: x+name_list[0] if isinstance(x, str) else x)
    result[neu] = result[neu].applymap(lambda x: x + name_list[1] if isinstance(x, str) else x)
    result[ub] = result[ub].applymap(lambda x: x + name_list[2] if isinstance(x, str) else x)

    length = result.iloc[-1].dropna().apply(lambda x: len(x)).max()

    result = result.applymap(lambda x: np.nan if isinstance(x, str) and len(x) != length else x)

    return result

size_ia = divide_percentile(size, iva_y, kospi_y, 30, 70, ['L','M','H'])

size_m = size_ia.copy()
size_m.index = pd.to_datetime(size_m.index)
size_m = size_m.resample('M').fillna(method='ffill',limit=12).reindex(monthly_rtn.loc['2000-06':].index).fillna(method='ffill',limit=12)

kospi = monthly[monthly.iloc[:,0]=='시장구분'].drop(monthly.columns[0], axis=1).T.replace('KS',True).replace(['EX','KQ'],[False, False]).astype('bool')
kospi.index = pd.to_datetime(kospi.index)

kospi_m = kospi.loc['2000-06':'2019-12']
kospi_m.index = size_m.index

size_ia_roe = divide_percentile(size_m, roe_m, kospi_m, 30, 70, ['L','M','H'])

size_ia_roe.dropna(how='all')


def get_rtn(x, result, monthly_rtn, market_cap):
    start_d = x.name.strftime('%Y-%m')
    end_d = x.name + pd.DateOffset(months=1)
    end_d = end_d.strftime('%Y-%m')
    unique_v = x.unique()

    for g in unique_v:
        tic_list = x[x == g].index
        tmp_mv = market_cap.loc[start_d,tic_list].iloc[0].dropna()
        tmp_rtn = monthly_rtn.loc[end_d, tic_list].iloc[0].dropna()
        tic = list(set(tmp_mv.index) & set(tmp_rtn.index))
        tic.sort()
        tmp_rtn = tmp_rtn.loc[tic]
        tmp_mv = tmp_mv.loc[tic]

        tmp_mv /= tmp_mv.sum()
        result.loc[end_d,g] = (tmp_rtn*tmp_mv).sum()

    return 'Done!'



def fac_port_rtn(group, monthly_rtn, market_cap):
    stock_list = group.dropna(how='all')
    port_rtn = pd.DataFrame(index=monthly_rtn.index, columns=stock_list.iloc[0].dropna().unique())
    tmp_df = stock_list.iloc[:-1].apply(lambda x: get_rtn(x.dropna(), port_rtn, monthly_rtn, market_cap), axis=1)

    return port_rtn.dropna(axis=0)

qfactor_port = fac_port_rtn(size_ia_roe,monthly_rtn,market_cap).loc['2003-01':]

small = qfactor_port[[x for x in qfactor_port.columns if x[0] == 'S']]
big = qfactor_port[[x for x in qfactor_port.columns if x[0] == 'B']]
low_inv = qfactor_port[[x for x in qfactor_port.columns if x[1] == 'L']]
high_inv = qfactor_port[[x for x in qfactor_port.columns if x[1] == 'H']]
low_roe = qfactor_port[[x for x in qfactor_port.columns if x[2] == 'L']]
high_roe = qfactor_port[[x for x in qfactor_port.columns if x[2] == 'H']]

smb_qfac = (small.mean(axis=1) - big.mean(axis=1))
inv_qfac = (low_inv.mean(axis=1) - high_inv.mean(axis=1))
roe_qfac = (high_roe.mean(axis=1) - low_roe.mean(axis=1))

mkt_fac = mkt_factor.loc['2003':]
mkt_fac.index = smb_qfac.index

rf_m = rf_m.loc['2003':'2020']
rf_m.index = smb_qfac.index

qfactor = pd.concat([mkt_fac, smb_qfac, inv_qfac, roe_qfac, rf_m],axis=1)
qfactor.columns = ['MKT-Rf', 'SMB(HXZ4)', 'IVA', 'ROE', 'Rf']

qfactor.to_csv('./factor construction/qfactor.csv', encoding='cp949')
qfactor_port.to_csv('./factor construction/qfactor_2x3x3.csv', encoding='cp949')