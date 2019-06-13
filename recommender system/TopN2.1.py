# INDEX=(TIME+COLLECTION+CLICK)/(TIMENOW-WEDDINGDATE)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import pymssql as ps
from sqlalchemy import create_engine,VARCHAR
import math
import pymysql
# conn1 = pymysql.connect(host='115.29.45.77', user='root', password='123456', charset='utf8',db='recommend_system',port=3306)

#---------------------处理异常值函数----------------------------
#三西格玛法：假设样本符合正态分布，则99%的数据应该处于均值的3个标准差之内
#箱型法：IQR(差值) = U(上四分位数) - L(下四分位数)
# 上界 = U + 1.5IQR
# 下界 = L-1.5IQR
#绝对中位差：样本-中位数的绝对值
# 上界 = 中位数+3*绝对中位差
# 下界 = 中位数-3*绝对中位差
def abdata_processing(data, column_names, method = 'absmedian'):
    new_data = data.copy()
    column_names_old = new_data.columns
    if method == 'sigma':
        for c in column_names:
            new_data[c+'_zscore'] = abs((new_data[c] - new_data[c].mean())/new_data[c].std())
            new_data[c] = new_data[new_data[c+'_zscore'] <= 3][c]
    elif method == 'boxplot':
        for c in column_names:
            new_data[c + '_ubound'] = new_data[c].quantile(0.75) + \
                                      1.5 * (new_data[c].quantile(0.75) - new_data[c].quantile(0.25))
            new_data[c + '_lbound'] = new_data[c].quantile(0.25) - \
                                      1.5 * (new_data[c].quantile(0.75) - new_data[c].quantile(0.25))
        for c in column_names:
            new_data = new_data[new_data[c] <= new_data[c + '_ubound']]
            new_data = new_data[new_data[c] >= new_data[c + '_lbound']]
    elif method == 'absmedian':
        b = 1.4826  # 这个值应该是看需求加的，有点类似加大波动范围之类的
        for c in column_names:
            new_data[c + '_ubound'] = pd.np.median(new_data[c]) + 3 * b * pd.np.median(
                abs(new_data[c] - pd.np.median(new_data[c])))
            new_data[c + '_lbound'] = pd.np.median(new_data[c]) - 3 * b * pd.np.median(abs(
                new_data[c] - pd.np.median(new_data[c])))
        for c in column_names:
            new_data = new_data[new_data[c] <= new_data[c + '_ubound']]
            new_data = new_data[new_data[c] >= new_data[c + '_lbound']]

    return new_data[column_names_old]

def d1tod2(data):
    x = []
    for i in range(0,len(data)):
        x.append([data.iloc[i]])
    return x

conn = ps.connect(host='10.66.184.119',user='test',password='tes@t123456#pwd2018&',charset='utf8')
# 浏览起始时间
tstart = pd.Timestamp.now()-pd.Timedelta('15 days')

view_start_time = str('\'%s\''%tstart.date())

sql_caseview = '''select 
a.case_id,
a.access_time,
a.user_id,
c.province
from tb_int_TopN_BaseData a
inner join tb_user b on a.user_id = b.id
left join tb_case c on a.case_id = c.number
where b.type = 0 and a.user_id is not null and a.access_time >= CONVERT(datetime, %s, 101)
and a.channel not in ('1thlook' ,'zhaowo-android' , 'zhaowo-ios') and c.status = 0 and c.creator_vocation_name='策划师';'''%view_start_time

sql_collect = '''select
distinct c.number case_id,
count(c.number) case_col_cnt
from tb_user_case_collection a
inner join tb_user b on a.user_id = b.id
left join tb_case c on a.case_id = c.id
where b.type = 0 and a.user_id is not null and a.ctime >= CONVERT(datetime, %s, 101) and a.status=0
and c.creator_vocation_name = '策划师' and c.status = 0
group by c.number;'''%view_start_time

sql_wd = '''select
distinct a.number case_id,b.wedding_date
from tb_case a left join tb_wedding_case b on a.id = b.case_id
where a.creator_vocation_name = '策划师' and a.status = 0;
'''

data_viewtime = pd.read_sql(sql_caseview,con=conn,parse_dates=['access_time'])
data_viewtime = data_viewtime.sort_values(by=['user_id','access_time'])
# 剔除数据异常案例
#print (data_viewtime.head(10))
data_viewtime = data_viewtime[data_viewtime['case_id']!=110001144]
# 计算浏览时长
data_viewtime['access_date'] = data_viewtime['access_time'].map(lambda x:pd.datetime.date(x))
data_viewtime[['access_time_next','user_id_next','access_date_next']] = data_viewtime[['access_time','user_id','access_date']].shift(-1)
data_viewtime['viewtime'] = data_viewtime['access_time_next'] - data_viewtime['access_time']
data_viewtime['viewtime'] = data_viewtime['viewtime'].map(lambda x:x.seconds)
data_viewtime['viewtime'][data_viewtime['user_id_next']!=data_viewtime['user_id']] = 0
data_viewtime['viewtime'][data_viewtime['access_date_next']!=data_viewtime['access_date']] = 0
data_viewtime = abdata_processing(data_viewtime,['viewtime'])
data_viewtime['click_cnt'] = 1
data_viewtime['user_cnt'] = data_viewtime['case_id'].map(lambda x:len(data_viewtime[data_viewtime['case_id']==x]['user_id'].drop_duplicates()))
#print (data_viewtime[['user_id', 'user_cnt', 'case_id']].head(20))

# 分案例汇总浏览时长
p_viewtime = pd.pivot_table(data_viewtime,index=['case_id'],values=['viewtime'],aggfunc=pd.np.sum)
#print (p_viewtime.head())
p_viewtime = p_viewtime.reset_index()
# 分案例汇总用户数
p_user = pd.pivot_table(data_viewtime,index=['case_id'],values=['user_cnt'],aggfunc=pd.np.average)
p_user = p_user.reset_index()
# 分案例汇总点击数
p_click = pd.pivot_table(data_viewtime,index=['case_id'],values=['click_cnt'],aggfunc=pd.np.sum)
p_click = p_click.reset_index()
# 案例对应的省
data_province = data_viewtime[['case_id','province']].drop_duplicates()
#print (data_province.head())

# 案例收藏数
data_collection = pd.read_sql(sql_collect,con=conn)
#print (data_collection.head(20))
# 案例对应婚期
data_wd = pd.read_sql(sql_wd,con=conn,parse_dates=['wedding_date'])

data = pd.merge(p_click,p_user,how='left',on='case_id')
data = pd.merge(data,p_viewtime,how='left',on='case_id')
data = pd.merge(data,data_collection,how='left',on='case_id')
data = pd.merge(data,data_wd,how='left',on='case_id')
data = pd.merge(data,data_province,how='left',on='case_id')
#print (data.head(20))

TOTAL_AVERAGE_VIEWTIME = data['viewtime'].sum()/data['user_cnt'].sum()
AVERAGE_USER_CNT = data['user_cnt'].sum()/len(data['user_cnt'])
TOTAL_AVERAGE_CLICK = data['click_cnt'].sum() / data['user_cnt'].sum()
# 平均案例浏览时长
data['self_average_viewtime'] = data['viewtime'] / data['user_cnt']
data['average_viewtime'] = data['self_average_viewtime'] * data['user_cnt'] / (data['user_cnt'] + AVERAGE_USER_CNT) \
                          + TOTAL_AVERAGE_VIEWTIME * AVERAGE_USER_CNT / (data['user_cnt'] + AVERAGE_USER_CNT)
# 平均案例点击数
data['self_average_click'] = data['click_cnt'] / data['user_cnt']
data['average_click'] = data['self_average_click'] * data['user_cnt'] / (data['user_cnt'] + AVERAGE_USER_CNT) \
                          + TOTAL_AVERAGE_CLICK * AVERAGE_USER_CNT / (data['user_cnt'] + AVERAGE_USER_CNT)
# 婚期距当前时间差
data['wedding_time'] = pd.Timestamp.now() - data['wedding_date']
#print (data['wedding_date'].sort_values(ascending=False).head(10))
#print (data['wedding_time'].sort_values().head(10))
data['wedding_time'] = data['wedding_time'].map(lambda x:x.days)
data = data.fillna(0)

scaler = MinMaxScaler()
data['case_col_cnt_score'] = scaler.fit_transform(d1tod2(data['case_col_cnt']))
data['average_viewtime_score'] = scaler.fit_transform(d1tod2(data['average_viewtime']))
data['average_click_score'] = scaler.fit_transform(d1tod2(data['average_click']))
data['res'] = data['case_col_cnt_score']+data['average_viewtime_score']+data['average_click_score']

# 时间系数，保证90天之前的婚礼很难进入前90名
#print (len(data))
r1 = data['res'].max()/data['res'].quantile((len(data)-90)/len(data))
#r2 = math.log(r1,90)
r2 = 1 # time period changed from 180d to 15d, so no need to consider aging now
data['wedding_time_score'] = data['wedding_time'].map(lambda x:math.pow(x,r2))

# 得到排名
data['sot'] = data['res']/data['wedding_time_score']
data = data.sort_values(by=['sot'])
# 只留近半年婚礼
data = data[data['wedding_date']>=view_start_time]
data['ctime'] = pd.datetime.now().date()
data['index'] = range(0,len(data))
# data.to_excel(r'D:\izhaowo\20181025 订单分配机制\topN案例\第二版\上线数据测试0313.xlsx',encoding='gbk',index=False)
# yconnect = create_engine('mysql+pymysql://root:123456@115.29.45.77:3306/recommend_system?charset=utf8')
yconnect = create_engine('mysql+pymysql://root:123456@10.80.185.36:3306/recommend_system?charset=utf8')
pd.io.sql.to_sql(data, 'user_case_interact', yconnect, schema='recommend_system', if_exists='replace',index=False,dtype={'province':VARCHAR(length=11)})
pd.io.sql.to_sql(data, 'user_case_interact_log', yconnect, schema='recommend_system', if_exists='append',index=False,dtype={'province':VARCHAR(length=11)})
