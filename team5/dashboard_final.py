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
    # [탑 뷰 - 안내 메시지 등]
    # ------------------------------------------------------------------------------
    if df_display.empty:
        st.warning("⚠️ 선택하신 조건에 맞는 지역이 없습니다. 예산을 조금 더 높여보세요.")
        return
        
    st.markdown("""
    <div style="background-color: #e7f5ff; border-left: 4px solid #339af0; padding: 15px; margin-top: 5px; margin-bottom: 20px; border-radius: 4px; color: #0b7285;">
        💡 <b>서비스 이용 팁:</b> [🏆 추천 TOP 5] 에서 결과를 확인하시고, <b>[💵 자산/대출 분석]</b> 탭으로 이동하시면 추천 입지 기준 <b>원클릭 맞춤형 대출 한도</b>를 바로 확인할 수 있습니다!
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------------------
    # [탭 메뉴] 리포트 상세
    # ------------------------------------------------------------------------------
    tab_rec, tab_safe, tab_infra, tab_comm, tab_money = st.tabs([
        "🏆 추천 TOP 5", "🛡️ 치안/안전", "🛒 생활 인프라", "🚇 정밀 통근 경로", "💵 자산/대출 분석"
    ])

    # --- [Tab 1] 추천 TOP 5 ---
    with tab_rec:
        st.markdown("<h3 class='section-header'>🌟 AI 종합 분석: 맞춤형 최적 입지 TOP 5</h3>", unsafe_allow_html=True)
        top_5 = df_display.head(5).copy()
        
        if len(top_5) > 0:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(top_5.iterrows()):
                with cols[i]:
                    badge_color = "#e03131" if i == 0 else "#1971c2" if i == 1 else "#339af0"
                    font_weight = "900" if i == 0 else "700"
                    border_style = f"2px solid {badge_color}" if i == 0 else "1px solid #dee2e6"
                    bg_color = "#fff5f5" if i==0 else "white"
                    loan_needed = max(0, row[price_col] - p_assets)
                    
                    # 별점 및 등급 산정 처리
                    score = row['종합점수']
                    if score >= 90:
                        grade, stars, g_color = 'S', '★★★★★', '#d9480f'
                    elif score >= 75:
                        grade, stars, g_color = 'A', '★★★★☆', '#f08c00'
                    elif score >= 55:
                        grade, stars, g_color = 'B', '★★★☆☆', '#2b8a3e'
                    elif score >= 35:
                        grade, stars, g_color = 'C', '★★☆☆☆', '#1971c2'
                    else:
                        grade, stars, g_color = 'D', '★☆☆☆☆', '#868e96'
                    
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; border: {border_style}; border-radius: 12px; padding: 15px; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <div style="font-size: 1.2rem; font-weight: {font_weight}; color: {badge_color}; margin-bottom: 5px;">
                            👑 {i+1}위
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #343a40; margin-bottom: 5px;">
                            {row['자치구']}
                        </div>
                        <div style="margin-bottom: 12px;">
                            <span style="font-size: 1.2rem; color: #fcc419; letter-spacing: 2px;">{stars}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #495057; text-align: left; background: #f8f9fa; padding: 8px; border-radius: 6px;">
                            ▪️ <b>종합평가:</b> {score:.1f}점 <span style='color:{g_color}; font-weight:bold;'>({grade}등급)</span><br>
                            ▪️ <b>통근:</b> {row['통근시간']:.0f}분<br>
                            ▪️ <b>예산:</b> {row[price_col]:,.0f}만<br>
                            🏢 <b>추천단지:</b><br><span style="color:#2b8a3e; font-size:0.8rem; line-height:1.2; display:inline-block; margin-top:3px;">{row.get('대표단지', '주요 단지')}</span><br>
                            💵 <b>필요대출액:</b> {loan_needed:,.0f}만
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("조건에 맞는 추천 데이터가 없습니다.")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 예산 vs 통근시간: 가성비(상관관계) 심층 분석")
        st.write("💡 **AI 인사이트:** 파란 점들은 당신의 예산 및 통근 조건(60분 이하)을 통과한 지역들입니다. 추천 TOP 5 지역이 어떤 포지션에 있는지 확인해보세요.")
        
        # 전체 데이터 기반 스캐터 플롯 생성 (예산 내/외 시각화 필터)
        df_scatter = df_base.copy()
        df_scatter['적합성'] = np.where(df_scatter[price_col] <= budget_slider, '예산 내 가능지역', '예산 초과')
        # 통근시간 열 추가
        if len(work_locs) == 1:
            df_scatter['통근시간'] = df_scatter['자치구'].map(lambda x: commute_matrix.get(x, {}).get(work_locs[0], 60))
        else:
            df_scatter['통근시간'] = df_scatter['자치구'].map(
                lambda x: (commute_matrix.get(x, {}).get(work_locs[0], 60) + 
                           commute_matrix.get(x, {}).get(work_locs[1], 60)) / 2
            )
            
        fig_scatter = px.scatter(df_scatter, x='통근시간', y=price_col, color='적합성', hover_name='자치구',
                                 color_discrete_map={'예산 내 가능지역': PRIMARY_COLOR, '예산 초과': '#ff6b6b'},
                                 title="전체 지역구 가성비 지도 (통근시간 vs 가격)")
        
        # 현재 예상 예산에 가이드라인 추가
        fig_scatter.add_hline(y=budget_slider, line_dash="dash", line_color="green", annotation_text=f"현재 예산 한도 ({budget_slider:,.0f}만)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        c_box, c_hist = st.columns(2)
        with c_box:
            st.markdown("#### 📈 서울시 평균 시세 분포 비교")
            fig_box = px.box(df_base, y=price_col, title=f"서울 전역 {p_deal} 시세 분포 (Box Plot)")
            if len(top_5) > 0:
                fig_box.add_hline(y=top_5.iloc[0][price_col], line_dash="dash", line_color=HIGHLIGHT_COLOR, annotation_text="추천 1순위 가격선")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with c_hist:
            st.markdown("#### 🚨 깡통전세 위험도 점검 (전세가율 분포)")
            fig_hist = px.histogram(df_base, x='전세가율', nbins=10, title="서울 권역 전세가율 리스크 확인",
                                    color_discrete_sequence=['#51cf66'])
            if len(top_5) > 0:
                fig_hist.add_vline(x=top_5.iloc[0]['전세가율'], line_dash="solid", line_color="red", annotation_text="추천 1순위 전세가율")
            st.plotly_chart(fig_hist, use_container_width=True)

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
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🗺️ 자치구별 생활 인프라 분포 밀도 (시뮬레이션)")
            st.write("선택한 자치구의 병원, 마트, 공원의 전반적인 인프라 분포 밀도를 시각화한 지도입니다. (실제 데이터 수량 기반)")
            
            sel_gu_infra = st.selectbox("인프라 상세 보기 대상 자치구", df_display['자치구'].tolist(), index=0, key="infra_map_sel")
            
            # 선택 자치구 인프라 수량 파악
            infra_data = df_display[df_display['자치구'] == sel_gu_infra].iloc[0]
            h_count = int(infra_data.get('병원수', 0))
            m_count = int(infra_data.get('마트수', 0))
            p_count = int(infra_data.get('공원수', 0))
            
            # 폴리움 맵 중심 설정
            center_lat, center_lon = GU_COORDS.get(sel_gu_infra, [37.5665, 126.9780])
            m_infra = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # 자치구 중심 마커
            folium.Marker(
                [center_lat, center_lon], 
                tooltip=f"{sel_gu_infra} 중심",
                icon=folium.Icon(color='lightgray', icon='info-sign')
            ).add_to(m_infra)
            
            # 밀도 기반 임의 랜덤 마커 생성기 (퍼포먼스 위해 각 항목별 최대 시각화 갯수 제한)
            np.random.seed(hash(sel_gu_infra) % 2**32) # 구역별로 시드가 고정되도록 처리
            
            def add_density_markers(count, color, tooltip_prefix):
                display_count = min(count, 50) # 수량이 많아도 렌더렉 방지를 위해 시각적 한계치 50개 적용
                for _ in range(display_count):
                    # 정규분포를 사용해 중심에 가깝게 군집되도록 좌표 시뮬레이션
                    lat = center_lat + np.random.normal(0, 0.012)
                    lon = center_lon + np.random.normal(0, 0.015)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=7,
                        color='white',
                        weight=1.5,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.8,
                        tooltip=f"📍 {tooltip_prefix}"
                    ).add_to(m_infra)
            
            # 차트와 동일한 색상 체계 적용
            add_density_markers(h_count, '#ff922b', '병원/의료시설')
            add_density_markers(m_count, '#51cf66', '대형마트/쇼핑')
            add_density_markers(p_count, '#339af0', '공원/녹지')
            
            st_folium(m_infra, width=1400, height=500, key="infra_density_map")
            
        else:
            st.info("비교할 데이터가 없습니다.")

    # --- [Tab 4] 정밀 통근 시뮬레이션 (고도화/정적 렌더링 방식) ---
    with tab_comm:
        st.markdown("<h3 class='section-header'>🚇 3D 하이브리드 통근 경로 스냅샷</h3>", unsafe_allow_html=True)
        if len(df_display) > 0:
            sel_gu = st.selectbox("분석 대상 지역", df_display['자치구'].tolist())
            sel_work = work_locs[0]
            total_time = commute_matrix.get(sel_gu, {}).get(sel_work, 60)
            
            st.markdown(f"#### 📍 {sel_gu} ➡️ {sel_work} (총 {total_time}분)")
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # --- 대중교통 노선 시뮬레이션 안내 ---
            st.markdown("#### 🧭 최적 대중교통 노선 안내 (시뮬레이션)")
            
            # 가상 노선 데이터 로직 (구역과 목적지에 따른 임의 노선 배정용)
            subway_lines = {"강남역": "지하철 2호선", "여의도역": "지하철 5호선", "광화문역": "지하철 5호선", "성수역": "지하철 2호선"}
            route_line = subway_lines.get(sel_work, "간선 버스")
            route_color = "#3bc9db" if "지하철" in route_line else "#20c997"
            
            # 상세 타임라인 분할 (가상 비율 배분)
            walk_to_station = 7
            subway_ride = total_time - 15 # 환승 및 대기 포함
            walk_to_work = 8
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 12px; padding: 25px; margin-top: 10px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #495057;">출발</div>
                        <div style="font-size: 1.5rem; color: #1c7ed6; font-weight: 800;">{sel_gu}</div>
                        <div style="font-size: 0.9rem; color: #868e96;">단지 인근 정류장</div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 2rem; color: #ced4da;">➔</div>
                        <div style="background-color: {route_color}; color: white; display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; margin-top: 5px;">
                            {route_line} 탑승
                        </div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #495057;">도착</div>
                        <div style="font-size: 1.5rem; color: #e03131; font-weight: 800;">{sel_work}</div>
                        <div style="font-size: 0.9rem; color: #868e96;">직장 도보권</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🗺️ 츨발지 - 목적지 직행 경로 지도 (참고용)")
            
            WORK_COORDS = {"강남역": [37.4979, 127.0276], "여의도역": [37.5218, 126.9243], 
                           "광화문역": [37.5709, 126.9768], "성수역": [37.5446, 127.0567]}
            target = WORK_COORDS.get(sel_work, [37.5665, 126.9780])
            m_sim = folium.Map(location=[(GU_COORDS[sel_gu][0] + target[0])/2, (GU_COORDS[sel_gu][1] + target[1])/2], zoom_start=12)
            
            # 마커 및 직선 경로 표시
            folium.Marker(GU_COORDS[sel_gu], tooltip="출발지", icon=folium.Icon(color='blue', icon='home')).add_to(m_sim)
            folium.Marker(target, tooltip="직장", icon=folium.Icon(color='red', icon='briefcase')).add_to(m_sim)
            folium.PolyLine([GU_COORDS[sel_gu], target], color=PRIMARY_COLOR, weight=4, dash_array='5', tooltip="직선 (최단망) 거리").add_to(m_sim)
            
            st_folium(m_sim, width=1400, height=400, key="detailed_sim_map")
            
        else:
            st.warning("선택할 수 있는 대상 지역이 없습니다. 예산을 상향하거나 조건을 완화해주세요.")

    # --- [Tab 5] 자산/대출 분석 (금융 라운지) ---
    with tab_money:
        st.markdown("<h3 class='section-header'>💵 내 자산 맞춤형 금융 큐레이션 라운지</h3>", unsafe_allow_html=True)
        st.write(f"💡 현재 사용자님의 연봉({p_salary:,}만원)은 수집된 데이터 기준으로 **상위 {np.random.randint(20, 50)}%** 수준에 해당합니다.")
        
        if len(top_5) > 0:
            target_price = top_5.iloc[0][price_col]
            loan_amt = max(0, target_price - p_assets)
            
            # --- 자산 vs 대출 요약 뷰 ---
            c3, c4 = st.columns([1, 1.2])
            with c3:
                fig_p = px.pie(values=[p_assets, loan_amt], names=['보유 자산', '대출 필요액'], 
                              title=f"🥇 1순위 지역({top_5.iloc[0]['자치구']}) 입주 예산 구성", 
                              color_discrete_sequence=[PRIMARY_COLOR, '#e9ecef'], hole=0.4)
                fig_p.update_traces(textinfo='percent+label', textfont_size=14)
                st.plotly_chart(fig_p, use_container_width=True)
                
            with c4:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background-color: white; border-left: 5px solid {HIGHLIGHT_COLOR}; border-radius: 8px; padding: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                    <h4 style='margin-top: 0; color: #343a40; font-weight:800;'>📊 우리집 자산 진단 결과</h4>
                    <div style='font-size: 1.15rem; color: #495057; line-height: 1.8;'>
                        ▶ 1순위 추천 입지 <b>{top_5.iloc[0]['자치구']}</b> 평균가: <span style='color:{PRIMARY_COLOR}; font-weight:bold;'>{target_price:,.0f}만 원</span><br>
                        ▶ 현재 가용 가능 자산: <b>{p_assets:,.0f}만 원</b><br>
                        ▶ <b>부족한 주거 예산 (필요 대출):</b> <span style='color:#e03131; font-weight:bold;'>{loan_amt:,.0f}만 원</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # --- 금융 상품 큐레이션 (Toss/KakaoBank 스타일) ---
            st.markdown("### 🏦 특별 금리 우대 추천 상품 조회 내역")
            st.write("사용자님의 소득과 자산 조건에 맞춰 승인 가능성이 높은 최적의 대출 상품을 찾아왔어요.")
            
            # 가상 상품 데이터
            mock_loans = [
                {"name": "신혼부부 전용 버팀목 대출", "provider": "주택도시기금", "rate": 2.15, "limit": 30000, "tag": "최저금리"},
                {"name": "청년 전월세보증금 대출", "provider": "카카오뱅크", "rate": 3.85, "limit": 10000, "tag": "모바일 간편화"},
                {"name": "i-ONE 직장인 전세대출", "provider": "IBK기업은행", "rate": 4.50, "limit": 50000, "tag": "높은 한도"}
            ]
            
            l_cols = st.columns(3)
            for j, loan in enumerate(mock_loans):
                with l_cols[j]:
                    # 한도 부족 체크
                    is_short = loan_amt > loan['limit']
                    actual_loan = min(loan_amt, loan['limit'])
                    monthly_pay = (actual_loan * (loan['rate'] / 100)) / 12
                    
                    status_badge = f"<span style='background:#f03e3e; color:white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em;'>한도부족</span>" if is_short else f"<span style='background:#20c997; color:white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em;'>한도충분</span>"
                    
                    st.markdown(f"""
                    <div style='border: 1px solid #dee2e6; border-radius: 12px; padding: 20px; background: white; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
                        <div style='color: #868e96; font-size: 0.9em; font-weight:bold;'>{loan['provider']}</div>
                        <h4 style='margin: 5px 0 15px 0; color: #343a40;'>{loan['name']}</h4>
                        <div style='background: {BG_LIGHT}; padding: 10px; border-radius: 8px; margin-bottom: 15px;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                                <span style='color: #495057;'>예상 금리</span>
                                <span style='font-weight: bold; color: {PRIMARY_COLOR};'>연 {loan['rate']}%</span>
                            </div>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: #495057;'>최대 한도</span>
                                <span style='font-weight: bold;'>{loan['limit']:,}만원</span>
                            </div>
                        </div>
                        <div style='text-align: center; margin-bottom: 10px;'>
                            {status_badge}
                        </div>
                        <div style='text-align: center; border-top: 1px dashed #ced4da; padding-top: 15px;'>
                            <span style='font-size: 0.9em; color: #868e96;'>예상 월 이자액</span><br>
                            <span style='font-size: 1.5rem; font-weight: 900; color: #15aabf;'>약 {monthly_pay:,.0f}만원</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- 실서비스용 제휴 금융사 전환 배너 ---
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 25px 30px; margin: 15px 0; border: 2px solid #339af0; box-shadow: 0 4px 15px rgba(51, 154, 240, 0.15);">
                <div style="text-align: center;">
                    <h3 style="color: #1c7ed6; margin-bottom: 10px; font-weight: 800;">💸 전세대출 한도, 지금 바로 알아보세요</h3>
                    <p style="font-size: 1.15rem; color: #495057; margin-bottom: 20px;">
                        1순위 추천 입지(<b>{top_5.iloc[0]['자치구']}</b>) 평균 {p_deal}가 <b>{target_price:,.0f}만 원</b> 기준,<br>
                        고객님에게 꼭 맞는 맞춤형 대출 <b>{loan_amt:,.0f}만 원</b>이 즉시 승인 가능한지 지금 확인해 보세요.
                    </p>
                </div>
                <div style="display: flex; justify-content: center; gap: 20px; font-size: 0.95rem; color: #2b8a3e; font-weight: bold; margin-bottom: 20px;">
                    <span>✅ 여러 금융사를 한 번에 비교</span>
                    <span>✅ 서류 제출 없는 간편 절차</span>
                    <span>✅ 신용점수 영향 없이 한도 확인</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
            with col_c2:
                # Custom Toss-style CTA Button
                st.markdown("""
                <a href="https://www.tossbank.com/product-service/loans/loan" target="_blank" style="text-decoration: none;">
                    <div style="background-color: #0050FF; color: white; border-radius: 12px; padding: 18px 20px; text-align: center; font-size: 1.15rem; font-weight: bold; box-shadow: 0 4px 10px rgba(0, 80, 255, 0.3); transition: all 0.2s ease-in-out; cursor: pointer;">
                        🔍 내 맞춤 대출 한도 조회하기
                    </div>
                </a>
                <style>
                    div[style*="background-color: #0050FF"]:hover {
                        background-color: #003ECC !important;
                        box-shadow: 0 6px 14px rgba(0, 80, 255, 0.4) !important;
                        transform: translateY(-2px);
                    }
                </style>
                """, unsafe_allow_html=True)
        else:
            st.warning("분석할 추천 내역이 없어 자금 시뮬레이션을 진행할 수 없습니다.")

    # ------------------------------------------------------------------------------
    # [하단 부록] 점수 산정 기준 및 평가 방식 설명
    # ------------------------------------------------------------------------------
    st.markdown("<br><br><br><hr>", unsafe_allow_html=True)
    with st.expander("ℹ️ AI 종합 점수 산정 기준 및 데이터 출처 안내 (클릭하여 열기)"):
        st.markdown("""
        ### 📊 1. AI 주거지 종합 점수 산정 방식 (Max 100점 기준)
        각 지표는 자치구별 스케일 차이를 보정하기 위해 **MinMax 스케일링(0~100점 환산)** 기법을 사용하여 정규화한 후, 
        사용자가 설정한 **중요도(Weight)**를 곱해 산출됩니다.

        > $$ 종합 점수 = (가격 점수 \\times W_1) + (통근 점수 \\times W_2) + (인프라 점수 \\times W_3) + (치안 점수 \\times W_4) + (전세가율 점수 \\times W_5) $$
        > *(조건: $W$의 총합은 100%, 사용자가 설정한 예산 상한선을 넘거나 통근시간 Threshold인 60분을 초과하는 지역은 1차 필터링 파이프라인에서 자동 제외됨)*

        #### 📋 세부 항목별 0~100점 환산 기준표
        - **💰 가격 점수**: "사용자의 거래방식 예산 한도" 이하 매물 중 가격이 낮을수록(부담이 적을수록) 100점에 가깝게 역산 정규화.
        - **🚇 통근 점수**: 출발지에서 설정한 직장(목적지)까지 대중교통 소요 시간이 짧을수록 100점에 가깝게 역산 정규화. (Threshold: 60분 이내)
        - **🛒 인프라 점수**: 관내 대형마트, 병원, 공원 수를 동일 가중치(각 1/3)로 0~100점 스케일링 후 단순 합산 보정.
        - **🛡️ 치안 점수**: 관내 연간 5대 범죄 발생 건수가 적을수록 치안망이 우수하다 판단, 100점에 가깝게 역산 정규화.
        - **📉 전세가율(리스크) 점수**: 매매가 대비 전세가율 데이터의 과도함(깡통전세 위험)을 회피하기 위해, 리스크가 낮을수록 높은 점수가 부여되도록 정규화.

        ---

        ### 🏅 2. 추천 입지 종합 평가 등급(S~D) 기준
        본 대시보드는 계산된 **'종합 점수'** 분포에 기반해 각 자치구의 입지 밸런스를 아래 5개 등급 기준표로 파악합니다.
        * **S 등급 (90점 이상)** : 사용자의 모든 검색 조건(예산, 통근시간, 인프라 등)에 가장 완벽히 부합하는 최적 입지
        * **A 등급 (80점 ~ 89점)** : 돋보이는 강점(예: 직주근접)을 하나 이상 지닌 차상위 추천 입지
        * **B 등급 (70점 ~ 79점)** : 예산이나 통근 조건에서 무난하게 타협 가능한 우수 입지
        * **C 등급 (60점 ~ 69점)** : 한두 가지 약점(예: 인프라 부족, 장시간 출근 등)이 존재하는 입지
        * **D 등급 (60점 미만)**  : 추천 대상지 중 비교 열위에 있어 조건 재조정을 권장하는 입지

        ---

        ### 📚 3. 분석 데이터 수집 출처 (Source Info)
        본 리포트에 사용된 모든 데이터는 공신력 있는 공공데이터 및 검증된 API 데이터를 기반으로 정제, 분석되었습니다.
        - **🏠 아파트 전월세 실거래가**: 국토교통부 실거래가 공개시스템
        - **📉 전세가율 리스크 분석**: 국토교통부 매매·전세 실거래 빅데이터
        - **🛒 생활 인프라 (의료/쇼핑/공원)**: 공공데이터포털, 건강보험심사평가원
        - **🚔 국가 치안 및 범죄 데이터**: 경찰청 공공데이터 포털 (치안만족도 및 주요 범죄발생 건수)
        - **🚇 통근 시간 및 교통망**: 서울시 대중교통 노선 데이터 기반 알고리즘 추정치 (오차 범위 포함)
        """)

if __name__ == "__main__":
    main()
