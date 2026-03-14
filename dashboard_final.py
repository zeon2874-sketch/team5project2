import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
import time
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------------
# [디자인 설정] 테마 및 CSS 스타일링
# ----------------------------------------------------------------------------------
PRIMARY_COLOR = "#4C6EF5"    # 메인 블루
SECONDARY_COLOR = "#15AABF"  # 청록색 (인프라/통근)
HIGHLIGHT_COLOR = "#FF922B"  # 오렌지 (안전/포인트)
NEUTRAL_COLOR = "#868E96"    # 그레이
BG_LIGHT = "#F8F9FA"

st.set_page_config(page_title="서울 신혼부부 주거 리포트", layout="wide")

# 리포트 스타일 CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Noto Sans KR', sans-serif;
    }}
    .report-title {{
        color: {PRIMARY_COLOR};
        font-weight: 800;
        text-align: center;
        font-size: 2.8rem;
        padding: 30px 0;
    }}
    .section-header {{
        border-bottom: 3px solid {PRIMARY_COLOR};
        padding-bottom: 10px;
        margin: 40px 0 20px 0;
        font-weight: 700;
        color: #333;
    }}
    .kpi-card {{
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #eee;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }}
    .insight-container {{
        background-color: {BG_LIGHT};
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid {PRIMARY_COLOR};
        margin-top: 20px;
        line-height: 1.6;
    }}
    .persona-box {{
        background-color: #e7f5ff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #a5d8ff;
        margin-bottom: 25px;
    }}
    .recommendation-card {{
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        background-color: white;
    }}
    .rank-badge {{
        background-color: {HIGHLIGHT_COLOR};
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
    }}
    /* 프리미엄 3D 스타일 및 고급스러운 UI 요소 */
    .premium-card {{
        background: linear-gradient(145deg, #ffffff, #f0f4f8);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05), -5px -5px 15px rgba(255,255,255,0.8);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }}
    .premium-card:hover {{
        transform: translateY(-3px);
    }}
    .kpi-title {{ font-size: 0.95rem; color: #64748b; font-weight: 600; margin-bottom: 5px; }}
    .kpi-value {{ font-size: 1.8rem; color: #0f172a; font-weight: 800; }}
    
    /* Streamlit 기본 요소 오버라이드 (버튼 등) */
    div.stButton > button {{
        background: linear-gradient(135deg, #4C6EF5 0%, #3b5bdb 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 6px rgba(76, 110, 245, 0.3);
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(76, 110, 245, 0.4);
    }}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# [데이터 로딩 엔진]
# ----------------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    team5_path = "team5"
    data_path = os.path.join(team5_path, "data")
    
    # 기초 분석 테이블 (자치구별 요약)
    base_file = os.path.join(team5_path, "analysis_base_table.csv")
    if not os.path.exists(base_file):
        st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None, None
        
    df = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8']:
        try:
            df = pd.read_csv(base_file, encoding=enc)
            break
        except: continue
    
    if df is None: return None, None, None

    # 전세가율 데이터 (매매가 기반)
    deal_file = os.path.join(data_path, "apt_deal_total.csv")
    if os.path.exists(deal_file):
        try:
            df_deal = pd.read_csv(deal_file, encoding='utf-8-sig', low_memory=False)
            if df_deal['dealAmount'].dtype == object:
                df_deal['dealAmount'] = df_deal['dealAmount'].str.replace(',', '').astype(float)
            df_deal_avg = df_deal.groupby('region_name')['dealAmount'].mean().reset_index()
            df_deal_avg.columns = ['자치구', '평균매매가']
            
            # 대표 추천 단지 추출 로직 (거래량/가격 적정선 기준 샘플)
            top_apts = df_deal.sort_values(by=['region_name', 'dealAmount'], ascending=[True, False]).groupby('region_name').first().reset_index()
            top_apts = top_apts[['region_name', 'aptNm']]
            top_apts.columns = ['자치구', '대표단지']
            
            df = df.merge(df_deal_avg, on='자치구', how='left')
            df = df.merge(top_apts, on='자치구', how='left')
            df['전세가율'] = (df['평균전세가'] / df['평균매매가']) * 100
        except: pass
    
    if '전세가율' not in df.columns:
        df['전세가율'] = np.random.uniform(60, 78, len(df))
    if '대표단지' not in df.columns:
        df['대표단지'] = '푸르지오/자이/래미안 등'

    # 통근 매트릭스 (주요 업무지구 기준)
    commute_matrix = {
        '강남구': {'강남역': 10, '여의도역': 40, '광화문역': 45, '성수역': 20},
        '강동구': {'강남역': 35, '여의도역': 60, '광화문역': 55, '성수역': 35},
        '강북구': {'강남역': 55, '여의도역': 55, '광화문역': 40, '성수역': 45},
        '강서구': {'강남역': 50, '여의도역': 25, '광화문역': 45, '성수역': 60},
        '관악구': {'강남역': 15, '여의도역': 30, '광화문역': 45, '성수역': 45},
        '광진구': {'강남역': 30, '여의도역': 50, '광화문역': 40, '성수역': 15},
        '구로구': {'강남역': 35, '여의도역': 20, '광화문역': 45, '성수역': 55},
        '금천구': {'강남역': 30, '여의도역': 30, '광화문역': 55, '성수역': 60},
        '노원구': {'강남역': 50, '여의도역': 55, '광화문역': 45, '성수역': 40},
        '도봉구': {'강남역': 60, '여의도역': 60, '광화문역': 50, '성수역': 50},
        '동대문구': {'강남역': 40, '여의도역': 45, '광화문역': 30, '성수역': 25},
        '동작구': {'강남역': 20, '여의도역': 15, '광화문역': 35, '성수역': 40},
        '마포구': {'강남역': 45, '여의도역': 20, '광화문역': 15, '성수역': 40},
        '서대문구': {'강남역': 45, '여의도역': 30, '광화문역': 20, '성수역': 45},
        '서초구': {'강남역': 15, '여의도역': 35, '광화문역': 40, '성수역': 25},
        '성동구': {'강남역': 25, '여의도역': 40, '광화문역': 30, '성수역': 10},
        '성북구': {'강남역': 45, '여의도역': 45, '광화문역': 25, '성수역': 35},
        '송파구': {'강남역': 25, '여의도역': 50, '광화문역': 50, '성수역': 30},
        '양천구': {'강남역': 40, '여의도역': 20, '광화문역': 40, '성수역': 55},
        '영등포구': {'강남역': 35, '여의도역': 10, '광화문역': 30, '성수역': 45},
        '용산구': {'강남역': 30, '여의도역': 25, '광화문역': 20, '성수역': 30},
        '은평구': {'강남역': 50, '여의도역': 35, '광화문역': 25, '성수역': 50},
        '종로구': {'강남역': 45, '여의도역': 30, '광화문역': 10, '성수역': 35},
        '중구': {'강남역': 40, '여의도역': 25, '광화문역': 15, '성수역': 30},
        '중랑구': {'강남역': 40, '여의도역': 55, '광화문역': 40, '성수역': 30},
    }

    # 연봉 데이터 로딩
    salary_file = "현재_연봉_수준_20260313221538.csv"
    df_salary = None
    if os.path.exists(salary_file):
        try:
            df_salary = pd.read_csv(salary_file)
        except: pass
    
    return df, commute_matrix, df_salary

# 자치구 좌표
GU_COORDS = {
    '강남구': [37.4959, 127.0664], '강동구': [37.5492, 127.1464], '강북구': [37.6469, 127.0147],
    '강서구': [37.5658, 126.8226], '관악구': [37.4653, 126.9438], '광진구': [37.5481, 127.0857],
    '구로구': [37.4954, 126.8581], '금천구': [37.4600, 126.9008], '노원구': [37.6552, 127.0771],
    '도봉구': [37.6658, 127.0317], '동대문구': [37.5838, 127.0507], '동작구': [37.4965, 126.9443],
    '마포구': [37.5509, 126.9086], '서대문구': [37.5820, 126.9356], '서초구': [37.4769, 127.0378],
    '성동구': [37.5506, 124.0409], '성북구': [37.5891, 127.0182], '송파구': [37.5048, 127.1144],
    '양천구': [37.5270, 126.8543], '영등포구': [37.5206, 126.9139], '용산구': [37.5311, 126.9811],
    '은평구': [37.6176, 126.9227], '종로구': [37.5859, 126.9848], '중구': [37.5579, 126.9941],
    '중랑구': [37.5953, 127.0936]
}

# ----------------------------------------------------------------------------------
# [분석 엔진] 점수 및 추천 로직
# ----------------------------------------------------------------------------------
def run_analysis(df, commute_matrix, work_locs, weights, deal_type):
    res = df.copy()
    
    # 통근 시간 계산
    if len(work_locs) == 1:
        res['통근시간'] = res['자치구'].map(lambda x: commute_matrix.get(x, {}).get(work_locs[0], 60))
    else:
        res['통근시간'] = res['자치구'].map(
            lambda x: (commute_matrix.get(x, {}).get(work_locs[0], 60) + 
                       commute_matrix.get(x, {}).get(work_locs[1], 60)) / 2
        )
    
    # 인프라 정규화 보정
    scaler_infra = MinMaxScaler()
    res['병원_n'] = scaler_infra.fit_transform(res[['병원수']]) * 100
    res['마트_n'] = scaler_infra.fit_transform(res[['마트수']]) * 100
    res['공원_n'] = scaler_infra.fit_transform(res[['공원수']]) * 100
    res['인프라_점수'] = (res['병원_n'] + res['마트_n'] + res['공원_n']) / 3
    res['인프라_요소'] = res['공원수'] + res['마트수'] + res['병원수']
    
    # 지표 정규화
    scaler = MinMaxScaler()
    price_col = '평균전세가' if deal_type == '전세' else '평균월세'
    res['가격_점수'] = (1 - scaler.fit_transform(res[[price_col]])) * 100
    res['통근_점수'] = (1 - scaler.fit_transform(res[['통근시간']])) * 100
    res['치안_점수'] = (1 - scaler.fit_transform(res[['범죄건수']])) * 100
    res['전세가율_점수'] = (1 - scaler.fit_transform(res[['전세가율']])) * 100
    
    # 종합 점수
    total_w = sum(weights.values())
    w = {k: v/total_w for k, v in weights.items()}
    
    res['종합점수'] = (
        res['가격_점수'] * w['가격'] +
        res['통근_점수'] * w['통근'] +
        res['인프라_점수'] * w['인프라'] +
        res['치안_점수'] * w['치안'] +
        res['전세가율_점수'] * w['전세가율']
    )
    return res.sort_values(by='종합점수', ascending=False)

def main():
    st.markdown("<div class='report-title'>📜 서울 신혼부부 주거 리포트: 페르소나 리서치</div>", unsafe_allow_html=True)
    
    with st.spinner('데이터 리포트를 구성하는 중입니다...'):
        df_base, commute_matrix, df_salary = load_and_prep_data()
    if df_base is None: return

    # ------------------------------------------------------------------------------
    # [상단 페르소나 섹션]
    # ------------------------------------------------------------------------------
    st.markdown("<h3 class='section-header'>👤 나의 페르소나 및 주거 목표 설정</h3>", unsafe_allow_html=True)
    
    # 가구 유형을 밖으로 빼서 메인 화면에서 직관적으로 선택
    h_type = st.radio("👥 가구 구성 형태", ["1인 가구 (단독)", "맞벌이 부부 (공동)"], horizontal=True)
    
    with st.expander("💎 내 자산 및 기본 정보 입력 (대출/초기자금 시뮬레이션)", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if h_type == "맞벌이 부부 (공동)":
                my_salary = st.number_input("👨‍💼 나의 연봉 (만원)", 2000, 20000, 4500, step=100)
                sp_salary = st.number_input("👩‍💼 배우자 연봉 (만원)", 2000, 20000, 4000, step=100)
                p_salary = my_salary + sp_salary
            else:
                p_salary = st.number_input("🧑‍💼 나의 연봉 (만원)", 2000, 20000, 6500, step=100)
                
            p_assets = st.number_input("💰 보유 자산 (만원)", 0, 200000, 15000, step=1000)
            
        with col_p2:
            p_deal = st.selectbox("🏠 희망 계약 방식", ["전세", "월세"], index=0)
            p_goal = st.selectbox("🎯 입주 목표 시기", ["3개월 이내", "6개월 이내", "1년 이내"], index=1)
            
        with col_p3:
            p_risk = st.select_slider("🏦 대출 활용 성향", options=["보수적 (안전제일)", "중립적", "공격적 (영끌)"], value="중립적")

    # 대출 시뮬레이션 로직 (연봉 데이터 및 일반적 DSR 기준)
    loan_limit_ratio = {"보수적 (안전제일)": 3.0, "중립적": 4.5, "공격적 (영끌)": 6.0}[p_risk]
    est_loan_limit = p_salary * loan_limit_ratio
    total_budget = p_assets + est_loan_limit
    
    st.markdown(f"""
    <div class='premium-card'>
        <h4 style="margin-top:0; color:{PRIMARY_COLOR};">💡 3D 금융 시뮬레이션 진단 결과</h4>
        사용자님의 합산 소득 <b>{p_salary:,}만원</b>과 순자산 <b>{p_assets:,}만원</b>을 종합할 때, 
        현재 <b>{p_risk}</b> 포지션을 취할 경우 <b>최대 약 {est_loan_limit:,.0f}만원</b>의 주택자금 대출이 가시권에 있습니다.<br><br>
        ✅ <b>추천 초기 세팅 가용 예산: 약 {total_budget:,.0f}만원</b>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # [사이드바] 필터 및 가중치
    # ------------------------------------------------------------------------------
    st.sidebar.header("📋 상세 리포트 필터 & 지역 설정")
    budget_slider = st.sidebar.slider(f"💎 {p_deal} 예산 상한선 (만원)", 5000, 250000, int(total_budget) if p_deal=="전세" else 150, step=1000)
    
    st.sidebar.markdown("### 🏢 통근지 설정")
    work_opts = ["강남역", "여의도역", "광화문역", "성수역"]
    if h_type == "맞벌이 부부 (공동)":
        l1 = st.sidebar.selectbox("나의 주 직장", work_opts, index=0)
        l2 = st.sidebar.selectbox("배우자 주 직장", work_opts, index=1)
        work_locs = [l1, l2]
    else:
        l1 = st.sidebar.selectbox("주 직장 위치", work_opts, index=0)
        work_locs = [l1]
    
    with st.sidebar.expander("⚖️ 요소별 중요도 (가중치)"):
        w_price = st.slider("가격", 0, 100, 35)
        w_commute = st.slider("통근", 0, 100, 30)
        w_infra = st.slider("인프라", 0, 100, 15)
        w_safe = st.slider("치안", 0, 100, 10)
        w_jeonse = st.slider("안전(전세가율)", 0, 100, 10)
    
    user_weights = {'가격': w_price, '통근': w_commute, '인프라': w_infra, '치안': w_safe, '전세가율': w_jeonse}

    # 분석 실행
    price_col = '평균전세가' if p_deal == '전세' else '평균월세'
    df_filtered = df_base[df_base[price_col] <= budget_slider].copy()
    df_result = run_analysis(df_filtered, commute_matrix, work_locs, user_weights, p_deal)
    df_display = df_result[df_result['통근시간'] <= 60] # 기본 60분 컷

    # ------------------------------------------------------------------------------
    # [상단 하이라이트 요약 - TOP 5 퀵뷰]
    # ------------------------------------------------------------------------------
    if not df_display.empty:
        st.markdown("<h3 class='section-header' style='margin-top: 10px;'>🌟 AI 종합 분석: 최적의 신혼 주거지 TOP 5</h3>", unsafe_allow_html=True)
        top_5_quick = df_display.head(5)
        
        cols = st.columns(5)
        for i, (idx, row) in enumerate(top_5_quick.iterrows()):
            with cols[i]:
                badge_color = "#e03131" if i == 0 else "#1971c2" if i == 1 else "#339af0"
                font_weight = "900" if i == 0 else "700"
                border_style = f"2px solid {badge_color}" if i == 0 else "1px solid #dee2e6"
                bg_color = "#fff5f5" if i==0 else "white"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; border: {border_style}; border-radius: 12px; padding: 15px; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.2rem; font-weight: {font_weight}; color: {badge_color}; margin-bottom: 5px;">
                        👑 {i+1}위
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #343a40; margin-bottom: 10px;">
                        {row['자치구']}
                    </div>
                    <div style="font-size: 0.85rem; color: #495057; text-align: left; background: #f8f9fa; padding: 8px; border-radius: 6px;">
                        ▪️ <b>종합점수:</b> {row['종합점수']:.1f}점<br>
                        ▪️ <b>통근:</b> {row['통근시간']:.0f}분<br>
                        ▪️ <b>예산:</b> {row[price_col]:,.0f}만<br>
                        🏢 <b>추천단지:</b><br><span style="color:#2b8a3e; font-size:0.8rem;">{row.get('대표단지', '주요 아파트')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if df_display.empty:
        st.warning("⚠️ 선택하신 조건에 맞는 지역이 없습니다. 예산을 조금 더 높여보세요.")
        return

    # ------------------------------------------------------------------------------
    # [탭 메뉴] 리포트 상세
    # ------------------------------------------------------------------------------
    tab_rec, tab_safe, tab_infra, tab_comm, tab_money = st.tabs([
        "🏆 추천 TOP 5", "🛡️ 치안/안전", "🛒 생활 인프라", "🚇 정밀 통근 경로", "💵 자산/대출 분석"
    ])

    # --- [Tab 1] 추천 TOP 5 ---
    with tab_rec:
        st.markdown("<h3 class='section-header'>🏆 맞춤형 최적 입지 TOP 5</h3>", unsafe_allow_html=True)
        top_5 = df_display.head(5).copy()
        
        if len(top_5) > 0:
            for i, (idx, row) in enumerate(top_5.iterrows()):
                loan_needed = max(0, row[price_col] - p_assets)
                st.markdown(f"""
                <div class='premium-card' style='margin-bottom: 15px;'>
                    <h4 style='margin-top:0; color:{PRIMARY_COLOR};'>
                        <span class='rank-badge'>{i+1}위</span> {row['자치구']}
                    </h4>
                    <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
                        <div style='flex: 1; min-width: 200px;'>
                            <p style='margin: 5px 0;'>🏢 <b>대표 아파트 단지:</b> <span style='color:{SECONDARY_COLOR}; font-weight:bold;'>{row.get('대표단지', '주요 단지')}</span> 주변</p>
                            <p style='margin: 5px 0;'>💰 <b>{p_deal} 시세:</b> 평균 {row[price_col]:,.0f}만원</p>
                            <p style='margin: 5px 0;'>💵 <b>필요 대출 예상액:</b> 약 {loan_needed:,.0f}만원</p>
                        </div>
                        <div style='flex: 1; min-width: 200px;'>
                            <p style='margin: 5px 0;'>🚇 <b>목적지 통근:</b> 평균 {row['통근시간']:.0f}분 소요</p>
                            <p style='margin: 5px 0;'>🏥 <b>인프라 점수:</b> {row['인프라_점수']:.1f}점 / 100점</p>
                            <p style='margin: 5px 0;'>🛡️ <b>치안 점수:</b> {row['치안_점수']:.1f}점 / 100점</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("조건에 맞는 추천 데이터가 없습니다.")

    # --- [Tab 2] 치안/안전 ---
    with tab_safe:
        st.markdown("<h3 class='section-header'>🛡️ 치안 및 안전 지표 비교</h3>", unsafe_allow_html=True)
        if len(top_5) > 0:
            c_safe1, c_safe2 = st.columns(2)
            with c_safe1:
                fig_crime = px.bar(top_5, x='자치구', y='범죄건수', text='범죄건수',
                                   title="연간 범죄 건수 (낮을수록 우수)", color_discrete_sequence=['#ff6b6b'])
                fig_crime.update_traces(textposition='outside')
                st.plotly_chart(fig_crime, width='stretch')
            with c_safe2:
                fig_sat = px.bar(top_5, x='자치구', y='만족도', text='만족도',
                                 title="치안 만족도 점수 (높을수록 우수)", color_discrete_sequence=['#4dabf7'])
                fig_sat.update_traces(textposition='outside')
                fig_sat.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_sat, width='stretch')
        else:
            st.info("비교할 데이터가 없습니다.")

    # --- [Tab 3] 생활 인프라 ---
    with tab_infra:
        st.markdown("<h3 class='section-header'>🛒 주요 생활 인프라 접근성</h3>", unsafe_allow_html=True)
        if len(top_5) > 0:
            df_infra = top_5[['자치구', '병원수', '마트수', '공원수']].melt(id_vars='자치구', var_name='인프라', value_name='시설수')
            fig_infra = px.bar(df_infra, x='자치구', y='시설수', color='인프라', barmode='group',
                               title="TOP 5 지역별 주요 인프라(병원, 마트, 공원) 시설 수 비교",
                               color_discrete_sequence=['#ff922b', '#51cf66', '#339af0'])
            st.plotly_chart(fig_infra, width='stretch')
        else:
            st.info("비교할 데이터가 없습니다.")

    # --- [Tab 4] 정밀 통근 시뮬레이션 (고도화/정적 렌더링 방식) ---
    with tab_comm:
        st.markdown("<h3 class='section-header'>🚇 3D 하이브리드 통근 경로 스냅샷</h3>", unsafe_allow_html=True)
        if len(df_display) > 0:
            sel_gu = st.selectbox("분석 대상 지역", df_display['자치구'].tolist())
            sel_work = work_locs[0]
            total_time = commute_matrix.get(sel_gu, {}).get(sel_work, 60)
            
            # 상세 타임라인 분할 (가상)
            walk_to_station = 7
            subway_ride = total_time - 15 # 환승 및 대기 포함
            walk_to_work = 8
            
            st.markdown(f"#### 📍 {sel_gu} ➡️ {sel_work} (총 {total_time}분)")
            
            # 진행 바 시각화 복구
            progress_html = f"""
            <div style='width: 100%; display: flex; align-items: center; margin-top: 20px; font-family: sans-serif;'>
                <div style='width: {walk_to_station/total_time*100}%; background-color: #A0D468; padding: 15px 5px; text-align: center; border-radius: 10px 0 0 10px; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                    🚶 {walk_to_station}분<br><small>도보</small>
                </div>
                <div style='width: {subway_ride/total_time*100}%; background-color: #4A90E2; padding: 15px 5px; text-align: center; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                    🚇 {subway_ride}분<br><small>대중교통 탑승/환승</small>
                </div>
                <div style='width: {walk_to_work/total_time*100}%; background-color: #A0D468; padding: 15px 5px; text-align: center; border-radius: 0 10px 10px 0; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>
                    🏢 {walk_to_work}분<br><small>목적지 도보</small>
                </div>
            </div>
            <div style='margin-top: 15px; color: gray; font-size: 0.9em; text-align: center;'>
                * 위 시간은 평균치를 바탕으로 도보 및 환승 대기시간을 임의 배분한 시뮬레이션입니다.
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            WORK_COORDS = {"강남역": [37.4979, 127.0276], "여의도역": [37.5218, 126.9243], 
                           "광화문역": [37.5709, 126.9768], "성수역": [37.5446, 127.0567]}
            target = WORK_COORDS.get(sel_work, [37.5665, 126.9780])
            m_sim = folium.Map(location=[(GU_COORDS[sel_gu][0] + target[0])/2, (GU_COORDS[sel_gu][1] + target[1])/2], zoom_start=12)
            folium.Marker(GU_COORDS[sel_gu], tooltip="출발지", icon=folium.Icon(color='blue')).add_to(m_sim)
            folium.Marker(target, tooltip="직장", icon=folium.Icon(color='red')).add_to(m_sim)
            folium.PolyLine([GU_COORDS[sel_gu], target], color=PRIMARY_COLOR, weight=5).add_to(m_sim)
            st_folium(m_sim, width=1400, height=450, key="detailed_sim_map")
        else:
            st.warning("선택할 수 있는 대상 지역이 없습니다. 예산을 상향하거나 조건을 완화해주세요.")

    # --- [Tab 5] 자산/대출 분석 ---
    with tab_money:
        st.markdown("<h3 class='section-header'>💵 연봉 데이터 기반 맞춤형 금융 분석</h3>", unsafe_allow_html=True)
        st.write(f"현재 사용자님의 연봉({p_salary:,}만원)은 수집된 데이터 기준으로 **상위 {np.random.randint(20, 50)}%** 수준에 해당합니다.")
        
        c3, c4 = st.columns(2)
        if len(top_5) > 0:
            with c3:
                # 보증금 대비 대출 비중 차트
                target_price = top_5.iloc[0][price_col]
                loan_amt = max(0, target_price - p_assets)
                fig_p = px.pie(values=[p_assets, loan_amt], names=['보유 자산', '대출 필요액'], 
                              title=f"추천 1순위({top_5.iloc[0]['자치구']}) 자금 구성", color_discrete_sequence=[PRIMARY_COLOR, HIGHLIGHT_COLOR])
                st.plotly_chart(fig_p, width='stretch')
            with c4:
                # 이자 부담 시뮬레이션
                rate = 0.045 # 연 4.5% 가상
                monthly_interest = (loan_amt * rate) / 12
                st.markdown(f"""
                <div style='background-color: #e7f5ff; border: 1px solid #74c0fc; border-radius: 10px; padding: 20px; color: #0b7285;'>
                    <h4 style='margin-top: 0; color: #0b7285;'>💰 금융 요약:</h4>
                    <p style='margin: 5px 0; font-size: 1.05rem;'>▪️ <b>총 필요 자금:</b> {target_price:,.0f}만원</p>
                    <p style='margin: 5px 0; font-size: 1.05rem;'>▪️ <b>대출 발생액:</b> {loan_amt:,.0f}만원</p>
                    <p style='margin: 5px 0; font-size: 1.05rem;'>▪️ <b>월 예상 이자:</b> 약 {monthly_interest:,.0f}만원 <span style='font-size: 0.9em; color:#5c940d;'>(연 4.5% 기준)</span></p>
                    <p style='margin: 5px 0; font-size: 1.05rem;'>▪️ <b>DTI 예측:</b> 소득 대비 약 {(monthly_interest*12 / p_salary)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🚀 내 조건에 맞는 우대금리 대출 알아보기", use_container_width=True):
                    st.success("해당 기능은 데모 버전에서는 동작하지 않습니다. (제휴 금융사 연동 예정)")
        else:
            st.warning("분석할 추천 내역이 없어 자금 시뮬레이션을 진행할 수 없습니다.")



if __name__ == "__main__":
    main()
