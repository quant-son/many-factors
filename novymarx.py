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

time_ind = [x for x in monthly_rtn.index if x.month == 6][2:]
y_index = [x for x in q_index if x.month == 6]
y_index2 = [x.strftime('%Y-%m') for x in y_index]

bm = pd.read_csv('./anomalies/bm.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all').shift(1).astype('float')
gpa = pd.read_csv('./anomalies/gpa.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all').shift(1).astype('float')

mom = pd.read_csv('./anomalies/r11.csv', encoding='cp949', index_col=0, parse_dates=True).dropna(how='all')
mom.index = pd.to_datetime(mom.index)


industry = monthly[monthly.iloc[:, 0] == 'WICS업종명(중)'].drop(monthly.columns[0], axis=1).replace('보험', np.nan).T
industry.index = pd.to_datetime(industry.index)
industry_m = industry.loc['2000-12':'2019-12']
industry = industry.loc[time_ind]


def industry_demean(x, fac):
    date = x.name
    ind_list = x.dropna()
    unique = ind_list.unique()

    for ind in unique:
        tic_list = ind_list[ind_list==ind].index
        fac.loc[date, tic_list] -= fac.loc[date, tic_list].mean()

    return 'Done!'

bm_demean = bm.copy()
gpa_demean = gpa.copy()
mom_demean = mom.copy()

result1 = industry.apply(lambda x: industry_demean(x, bm_demean), axis=1)
result2 = industry.apply(lambda x: industry_demean(x, gpa_demean), axis=1)
result3 = industry_m.apply(lambda x: industry_demean(x, mom_demean), axis=1)

me_y = market_cap_q.loc[y_index]

kospi_y = kospi.loc[y_index]
kospi_y.index = me_y.index

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

    result = result.applymap(lambda x: np.nan if isinstance(x, str) and len(x) == 1 else x)

    return result



size_bm = divide_percentile(size, bm_demean, 30, 70, ['G', 'N', 'V'])
size_gpa = divide_percentile(size, gpa_demean, 30, 70, ['U', 'M', 'P'])



size_m = market_cap.copy()
size_m[:] = ''

kospi_m = monthly[monthly.iloc[:,0]=='시장구분'].drop(monthly.columns[0], axis=1).T.replace('KS',True).replace(['EX','KQ'],[False, False]).astype('bool')
kospi_m.index = pd.to_datetime(kospi_m.index)

median_m = market_cap[kospi_m].median(axis=1)

size_m[market_cap.lt(median_m,axis=0)]='S'
size_m[market_cap.ge(median_m,axis=0)]='B'

size_mom = divide_percentile(size_m, mom_demean, 30, 70, ['D', 'M', 'U'])


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


size_bm_port = fac_port_rtn(size_bm, monthly_rtn, market_cap).loc['2003-01':]
size_gpa_port = fac_port_rtn(size_gpa, monthly_rtn, market_cap).loc['2003-01':]




def get_rtn(x, result, monthly_rtn, market_cap):
    start_d = x.name.strftime('%Y-%m')
    end_d = x.name + pd.DateOffset(months=1)
    end_d = end_d.strftime('%Y-%m')
    unique_v = x.unique()

    for g in unique_v:
        tic_list = x[x == g].index
        tic_list = list(set(tic_list) & set(monthly_rtn.loc[end_d].iloc[0].dropna().index))
        tic_list.sort()
        tmp_mv = market_cap.loc[start_d, tic_list].iloc[0].dropna()
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

size_mom_port = fac_port_rtn(size_mom,monthly_rtn,market_cap).loc['2003-01':]


size_bm_port.to_csv('./factor construction/bm_nm_2x3.csv', encoding='cp949')
size_gpa_port.to_csv('./factor construction/gpa_nm_2x3.csv', encoding='cp949')
size_mom_port.to_csv('./factor construction/mom_nm_2x3.csv', encoding='cp949')

bm_nm_fac = size_bm_port[['SV','BV']].mean(axis=1) - size_bm_port[['SG','BG']].mean(axis=1)
gpa_nm_fac = size_gpa_port[['SP','BP']].mean(axis=1) - size_gpa_port[['SU','BU']].mean(axis=1)
mom_nm_fac = size_mom_port[['SU','BU']].mean(axis=1) - size_mom_port[['SD','BD']].mean(axis=1)

bm_nm_fac.name = 'HML(NM4)'
gpa_nm_fac.name = 'PMU'
mom_nm_fac.name = 'MOM(NM4)'

mkt_fac = mkt_factor.loc['2003':]
mkt_fac.index = bm_nm_fac.index
mkt_fac.name = 'MKT-Rf'


rf_m.index = pd.to_datetime(rf_m.index)
rf_m = rf_m.loc['2003':'2019']
rf_m.index = bm_nm_fac.index
rf_m.name = 'Rf'

nm4_fac = pd.concat([mkt_fac, bm_nm_fac, gpa_nm_fac, mom_nm_fac, rf_m], axis=1)

nm4_fac.to_csv('./factor construction/nm4.csv',encoding='cp949')