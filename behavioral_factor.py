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


q_index = pd.DatetimeIndex([x for x in monthly_rtn.loc['2000-01':].index if x.month == 3 or x.month == 6 or x.month == 9 or x.month == 12])
market_cap_q = market_cap.loc[q_index]

kospi = monthly[monthly.iloc[:,0]=='시장구분'].drop(monthly.columns[0], axis=1).T.replace('KS',True).replace(['EX','KQ'],[False, False]).astype('bool')
kospi.index = pd.to_datetime(kospi.index)
kospi = kospi.resample('Q').apply(lambda x: x[-1]).loc['2000':'2019']
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


#FIN
nsi = pd.read_csv('./anomalies/nsi.csv', encoding='cp949', index_col=0).dropna(how='all')
csi = pd.read_csv('./anomalies/csi.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
csi = csi.resample('Q').apply(lambda x: x[-1])
csi = csi.loc['2001-06':'2019']
nsi = nsi.shift(1)
nsi.index = csi.index


csi_y = csi.loc[[x for x in csi.index if x.month==6]].iloc[1:]

csi_group = csi_y.copy().astype('str')
csi_group[:] = ''
kospi_y2 = kospi_y.copy().iloc[2:]
kospi_y2.index = csi_y.index
csi_sort = divide_percentile(csi_group, csi_y, kospi_y2, 20, 80, ['L','M','H'])

size
nsi_y = nsi.loc[csi_y.index].dropna(how='all')

nsi_group = nsi_y.copy().astype('str')
nsi_group[:] = ''

def divide_nsi(size, fac, kospi):
    result = size.astype('object').copy()

    repurchase = fac[fac < 0]
    issuing = fac[fac > 0]

    repu_median = repurchase[kospi & fac < 0].median()
    result[repurchase.gt(repu_median)] = 'M'
    result[repurchase.le(repu_median)] = 'L'

    crit_lb = fac[kospi & fac>0].apply(lambda x: np.nanpercentile(x, 30), axis=1)
    crit_ub = fac[kospi & fac>0].apply(lambda x: np.nanpercentile(x, 70), axis=1)

    lb = issuing.lt(crit_lb,axis=0)
    neu = issuing.ge(crit_lb, axis=0) & fac.lt(crit_ub, axis=0)
    ub = issuing.ge(crit_ub, axis=0)

    result[lb] = result[lb].applymap(lambda x: x+'M' if isinstance(x, str) else x)
    result[neu] = result[neu].applymap(lambda x: x + 'M' if isinstance(x, str) else x)
    result[ub] = result[ub].applymap(lambda x: x + 'H' if isinstance(x, str) else x)

    length = result.iloc[-1].dropna().apply(lambda x: len(x)).max()
    result = result.applymap(lambda x: np.nan if isinstance(x, str) and len(x) != length else x)

    return result

nsi_sort = divide_nsi(nsi_group, nsi_y, kospi_y2)

nsi_sort = nsi_sort.replace(np.nan, 'N')
csi_sort = csi_sort.replace(np.nan, 'N')

fin = nsi_sort+csi_sort
fin = fin.replace(['HH','HN','LL','NL','LN','NN'], ['H','H','L','L','L',np.nan]).replace(['HL','HM', 'MH', 'ML', 'MM', 'MN', 'NH', 'NL', 'NM'],'M')
size
size = size.iloc[2:].astype('object')
size.index = fin.index
size_fin = size + fin



def get_rtn(x, result, monthly_rtn, market_cap):
    start_d = str(x.name.year)+'-07'
    end_d = str(x.name.year+1)+'-06'
    if int(x.name.year) == 2019:
        end_d = '2019-12'

    unique_v = x.unique()
    for g in unique_v:
        tic_list = x[x == g].index
        tmp_mv = market_cap.loc[x.name.strftime('%Y-%m'),tic_list].iloc[0]
        tmp_rtn = monthly_rtn.loc[start_d:end_d, tic_list]
        tmp_mv /= tmp_mv.sum()
        result.loc[start_d:end_d,g] = (tmp_rtn*tmp_mv).sum(skipna=True, axis=1)

    return 'Done!'


def fac_port_rtn(group, monthly_rtn, market_cap):
    stock_list = group.dropna(how='all')
    port_rtn = pd.DataFrame(index=monthly_rtn.index, columns=stock_list.iloc[0].dropna().unique())
    tmp_df = stock_list.apply(lambda x: get_rtn(x.dropna(), port_rtn, monthly_rtn, market_cap), axis=1)

    return port_rtn.dropna(axis=0)

size_fin_port = fac_port_rtn(size_fin, monthly_rtn, market_cap).loc['2003-01':]
size_fin_port.to_csv('./factor construction/fin_2x3.csv',encoding='cp949')
fin_fac = size_fin_port[['SL','BL']].mean(axis=1) - size_fin_port[['SH','BH']].mean(axis=1)
fin_fac.name = 'FIN'
fin_fac

#PEAD

abr = pd.read_csv('./anomalies/abr.csv',encoding='cp949',index_col=0, parse_dates=True).loc[:"2019"]

size_m = market_cap.copy()
size_m[:] = ''

kospi_m = monthly[monthly.iloc[:,0]=='시장구분'].drop(monthly.columns[0], axis=1).T.replace('KS',True).replace(['EX','KQ'],[False, False]).astype('bool')
kospi_m.index = pd.to_datetime(kospi_m.index)

median_m = market_cap[kospi_m].median(axis=1)

size_m[market_cap.lt(median_m,axis=0)]='S'
size_m[market_cap.ge(median_m,axis=0)]='B'

size_m = size_m.loc["2001":]

size_pead = divide_percentile(size_m, abr, kospi_m, 20, 80, ['L','M','H'])


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

size_pead_port = fac_port_rtn(size_pead, monthly_rtn, market_cap).loc['2003-01':]
size_pead_port.to_csv('./factor construction/pead_2x3.csv',encoding='cp949')
pead_fac = size_pead_port[['SH','BH']].mean(axis=1) - size_pead_port[['SL','BL']].mean(axis=1)
pead_fac.name = 'PEAD'

pd.concat([fin_fac, pead_fac],axis=1).to_csv('./factor construction/behavioral_factor.csv',encoding='cp949')