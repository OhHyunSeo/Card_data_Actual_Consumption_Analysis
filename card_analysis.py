#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:44:18 2025

@author: oh

카드 소비 데이터 분석 : bc_card.txt : 원본 데이터 (2019.6 전체 소비 데이터의 일부)

1. 데이터 로드 및 전처리
2. 데이터프레임 형식으로 저장
3. 데이터 분석

1). 서울시 거주/비거주 고객의 소비 분석

    - 서울시 거주/비거주 고객 수 구하기
    - 총 소비액 구하기
    - 성별 소비액 구하기
    - 카드 이용 건수 구하기

2). 편의점 소비 정보 분석

    - 편의점 소비액 구하기
    - 강남구 편의점 소비액 분석
    
3). 서울시 거주/비거주 고객의 소비액 구하기

4). 거주지 소재 편의점 소비액 구하기
"""

import pandas as pd

# 1. 데이터 로드 및 전처리 & 데이터 프레임으로 저장
df = pd.read_table("bc_card_data/bc_card_out2020_03.txt")
df.head()
'''
   REG_YYMM  MEGA_CTY_NO MEGA_CTY_NM  CTY_RGN_NO  ... AGE_VAL  FLC       AMT   CNT
0    202003           11       서울특별시        1168  ...     20대    1   7927440  1089
1    202003           11       서울특별시        1129  ...     40대    4    274100    25
2    202003           11       서울특별시        1144  ...  60대 이상    5  34395725   808
3    202003           11       서울특별시        1126  ...     20대    1  31860800  3699
4    202003           11       서울특별시        1168  ...     50대    4   2546487    45

[5 rows x 23 columns]
'''
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1589494 entries, 0 to 1589493
Data columns (total 23 columns):
 #   Column             Non-Null Count    Dtype 
---  ------             --------------    ----- 
 0   REG_YYMM           1589494 non-null  int64 
 1   MEGA_CTY_NO        1589494 non-null  int64 
 2   MEGA_CTY_NM        1589494 non-null  object
 3   CTY_RGN_NO         1589494 non-null  int64 
 4   CTY_RGN_NM         1589494 non-null  object
 5   ADMI_CTY_NO        1589494 non-null  int64 
 6   ADMI_CTY_NM        1589494 non-null  object
 7   MAIN_BUZ_CODE      1589494 non-null  int64 
 8   MAIN_BUZ_DESC      1589494 non-null  object
 9   TP_GRP_NO          1589494 non-null  int64 
 10  TP_GRP_NM          1589494 non-null  object
 11  TP_BUZ_NO          1589494 non-null  int64 
 12  TP_BUZ_NM          1589494 non-null  object
 13  CSTMR_GUBUN        1589494 non-null  object
 14  CSTMR_MEGA_CTY_NO  1589494 non-null  int64 
 15  CSTMR_MEGA_CTY_NM  1589494 non-null  object
 16  CSTMR_CTY_RGN_NO   1589494 non-null  int64 
 17  CSTMR_CTY_RGN_NM   1589494 non-null  object
 18  SEX_CTGO_CD        1589494 non-null  int64 
 19  AGE_VAL            1589494 non-null  object
 20  FLC                1589494 non-null  int64 
 21  AMT                1589494 non-null  int64 
 22  CNT                1589494 non-null  int64 
dtypes: int64(13), object(10)
memory usage: 278.9+ MB
'''
df.to_csv("./bc_card_data/new_card_data.csv")

# csv 형태로 다시 저장
df_edit = pd.read_csv("./bc_card_data/new_card_data.csv")
df_edit.head()
# => 결측치는 없음

'''
1). 서울시 거주/비거주 고객의 소비 분석

    - 서울시 거주/비거주 고객 수 구하기
    - 총 소비액 구하기
    - 성별 소비액 구하기
    - 카드 이용 건수 구하기
'''
# 서울시 거주/비거주 고객수 구하기
df_edit.columns
seoul = (df['CSTMR_MEGA_CTY_NM']=="서울특별시").sum()
# 901072

total = df['CSTMR_MEGA_CTY_NM'].count()
# 1589494
non_seoul = total - seoul
print(non_seoul)

# 서울시 거주/ 비거주별 총 소비액 구하기 
# 서울시 거주 고객들의 총 소비액 구하기
seoul_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] == "서울특별시", 'AMT'].sum()
# 1385914569631
non_seoul_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] != "서울특별시", 'AMT'].sum()
# 1940899349900

# 성별 소비액 구하기
# -> 서울시 거주 고객의 성별 소비액 : 남: 여: 
seoul_gender_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] == "서울특별시"].groupby('SEX_CTGO_CD')['AMT'].sum()
# SEX_CTGO_CD
# 1    682678945342
# 2    703235624289
# Name: AMT, dtype: int64

    
# / 서울시 비거주 고객의 성별 소비액 : 남: 여:
non_seoul_gender_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] != "서울특별시"].groupby('SEX_CTGO_CD')['AMT'].sum()
'''
SEX_CTGO_CD
1    1015425463575
2     925473886325
Name: AMT, dtype: int64
'''

# 카드 이용 건수 구하기
seoul_car_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] == "서울특별시", 'CNT'].sum()
# 58970629

non_seoul_car_amt = df.loc[df['CSTMR_MEGA_CTY_NM'] != "서울특별시", 'CNT'].sum()
# 53351160

'''
2). 편의점 소비 정보 분석

    - 편의점 소비액 구하기
    - 강남구 편의점 소비액 분석
'''
# 편의점 소비액 구하기
(df["TP_BUZ_NM"] =="편 의 점").sum()

conv_sale = df.loc[df['TP_BUZ_NM'] == "편 의 점", 'AMT'].sum()
# 79987167291

(df['CSTMR_CTY_RGN_NM']=="강남구").sum()
# 49661

(df['CTY_RGN_NM']=="강남구").sum()

gangnam_conv_sale1 = df.loc[(df['CTY_RGN_NM'] == "강남구") & (df['TP_BUZ_NM'] == "편 의 점"), 'AMT'].sum()
# 8170947461

gangnam_conv_gender_amt = df.loc[
    (df['CTY_RGN_NM'] == "강남구") & (df['TP_BUZ_NM'] == "편 의 점")
].groupby('SEX_CTGO_CD')['AMT'].sum()
'''
SEX_CTGO_CD
1    4949915757
2    3221031704
Name: AMT, dtype: int64
'''

gangnam_conv_age_avg = df.loc[
    (df['CSTMR_CTY_RGN_NM'] == "강남구") & (df['TP_BUZ_NM'] == "편 의 점")
].groupby('AGE_VAL')['AMT'].mean()


gangnam_conv_age_avg = df.loc[
    (df['CSTMR_CTY_RGN_NM'] == "강남구") & (df['TP_BUZ_NM'] == "편 의 점")
].groupby('AGE_VAL').apply(lambda x: x['AMT'].sum() / x['CNT'].sum())
'''
AGE_VAL
20대       5971.717625
20세 미만    3389.616618
30대       7122.221337
40대       7823.119214
50대       7426.419335
60대 이상    8088.162995
dtype: float64
'''

(df['CTY_RGN_NM'] == df['CSTMR_CTY_RGN_NM']).count()

home_conv_sale = df.loc[
    (df['CTY_RGN_NM'] == df['CSTMR_CTY_RGN_NM']) & (df['TP_BUZ_NM'] == "편 의 점"), 'AMT'
].sum()

# 44186716261





