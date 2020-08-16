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


def divide_percentile(size, fac, lb, ub, name_list):
    result = size.astype('object').copy()
    crit_lb = fac.apply(lambda x: np.nanpercentile(x, lb), axis=1)
    crit_ub = fac.apply(lambda x: np.nanpercentile(x, ub), axis=1)

    lb = fac.lt(crit_lb,axis=0)
    neu = fac.ge(crit_lb, axis=0) & fac.lt(crit_ub, axis=0)
    ub = fac.ge(crit_ub, axis=0)

    result[lb] = result[lb].applymap(lambda x: x+name_list[0] if isinstance(x, str) else x)
    result[neu] = result[neu].applymap(lambda x: x + name_list[1] if isinstance(x, str) else x)
    result[ub] = result[ub].applymap(lambda x: x + name_list[2] if isinstance(x, str) else x)

    length = result.iloc[-1].dropna().apply(lambda x: len(x)).max()

    result = result.applymap(lambda x: np.nan if isinstance(x, str) and len(x) != length else x)

    return result


#MGMT
nsi = pd.read_csv('./anomalies/nsi.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all').loc[:'2019']
csi = pd.read_csv('./anomalies/csi.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
csi = csi.resample('Q').apply(lambda x: x[-1])
csi = csi.loc['2000':]
nsi.index = csi.loc['2001-06':'2019'].index

ta = pd.read_csv('./anomalies/ta.csv', encoding='cp949', index_col=0).dropna(how='all')
ta.index = csi.loc['2001-06':'2019'].index
noa = pd.read_csv('./anomalies/noa.csv', encoding='cp949', index_col=0).dropna(how='all')
noa.index = nsi.index
ag = pd.read_csv('./anomalies/ag.csv', encoding='cp949', index_col=0).dropna(how='all')
ag.index = nsi.index
iva = pd.read_csv('./anomalies/iva.csv', encoding='cp949', index_col=0).dropna(how='all')
iva.index = nsi.index

mgmt_list = [nsi.shift(1), csi, ta.shift(1), noa.shift(1), ag.shift(1), iva.shift(1)]
mgmt_list = [x.loc["2002":"2019"] for x in mgmt_list]
mgmt_rank = [x.rank(pct=True,axis=1) for x in mgmt_list]
mgmt_value = sum(mgmt_rank)/len(mgmt_list)
mgmt_y = mgmt_value.loc[[x for x in mgmt_value.index if x.month==6]]

size = size.iloc[2:]
size.index = mgmt_y.index
size_mgmt = divide_percentile(size, mgmt_y, 20, 80, ['L', 'N', 'H'])


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

size_mgmt_port = fac_port_rtn(size_mgmt,monthly_rtn,market_cap).loc['2003-01':]

low_mgmt = size_mgmt_port[[x for x in size_mgmt_port.columns if x[-1]=='L']].sum(axis=1)/2
high_mgmt = size_mgmt_port[[x for x in size_mgmt_port.columns if x[-1]=='H']].sum(axis=1)/2

mgmt_fac = low_mgmt-high_mgmt
mgmt_fac.name = 'MGMT'


#PERF
fp = pd.read_csv('./anomalies/fp.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
fp = fp.resample('Q').apply(lambda x: x[-1])
fp = fp.loc[csi.loc['2001-06':'2019-12'].index]

oscore = pd.read_csv('./anomalies/oscore.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
oscore.index = csi.loc['2001-06':'2019'].index
mom = pd.read_csv('./anomalies/r11.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
mom.index = pd.to_datetime(mom.index)
mom = mom.resample('Q').apply(lambda x: x[-1])
mom = mom.iloc[2:]
mom.index = nsi.loc[:"2019"].index



gpa = pd.read_csv('./anomalies/gpa.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
gpa.index = csi.loc['2001-06':'2019'].index
roa = pd.read_csv('./anomalies/roa.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
roa = roa.resample('Q').apply(lambda x: x[-1])
roa = roa.loc["2001-06":"2019"]
roa.index = csi.loc["2001-06":'2019'].index


perf_list = [fp, oscore.shift(1), mom, gpa.shift(1), roa]
perf_rank = [x.rank(pct=True,axis=1) for x in perf_list]
perf_value = sum(perf_rank)/len(perf_list)
perf_y = perf_value.loc[[x for x in perf_value.index if x.month==6]]

size_perf = divide_percentile(size, perf_y, 20, 80, ['L', 'N', 'H'])
size_perf_port = fac_port_rtn(size_perf,monthly_rtn,market_cap).loc['2003-01':]

low_perf = size_perf_port[[x for x in size_perf_port.columns if x[-1]=='L']].sum(axis=1)/2
high_perf = size_perf_port[[x for x in size_perf_port.columns if x[-1]=='H']].sum(axis=1)/2

perf_fac = low_perf-high_perf
perf_fac.name = 'PERF'


#SIZE

smb_mispricing = (size_mgmt_port['SN']+size_perf_port['SN'])/2-(size_mgmt_port['BN']+size_perf_port['BN'])/2
smb_mispricing.name = 'SMB(SY4)'

mkt_fac = mkt_factor.loc['2003':]
mkt_fac.index = smb_mispricing.index
mkt_fac.name = 'MKT-Rf'


rf_m.index = pd.to_datetime(rf_m.index)
rf_m = rf_m.loc['2003':'2019']
rf_m.index = smb_mispricing.index
rf_m.name = 'Rf'

mispricing_factor = pd.concat([mkt_fac, smb_mispricing, mgmt_fac, perf_fac, rf_m],axis=1)
mispricing_factor.to_csv('./factor construction/mispricing_fac.csv',encoding='cp949')

size_mgmt_port.to_csv('./factor construction/mgmt_2x3.csv',encoding='cp949')
size_perf_port.to_csv('./factor construction/perf_2x3.csv',encoding='cp949')
