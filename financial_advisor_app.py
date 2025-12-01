import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="WealthGenie | AI Financial Advisor",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 600;
        margin-top: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        height: 3rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- The Stock Universe (Proxy for "Top 100") ---
# A curated list of market movers across sectors to scan
STOCK_UNIVERSE = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'LLY', 'TSLA', 'AVGO',
    'JPM', 'V', 'WMT', 'XOM', 'MA', 'UNH', 'PG', 'JNJ', 'HD', 'MRK',
    'COST', 'ABBV', 'CVX', 'CRM', 'BAC', 'KO', 'PEP', 'AMD', 'NFLX', 'DIS',
    'MCD', 'CSCO', 'INTC', 'VZ', 'T', 'NKE', 'PFE', 'BA', 'GS', 'IBM',
    'CAT', 'HON', 'UNP', 'TXN', 'QCOM', 'LOW', 'SPGI', 'AXP', 'RTX', 'BLK'
]

# --- Helper Functions ---

def fetch_stock_data(tickers):
    """Fetches real-time data for the stock universe."""
    data = []
    # Using a progress bar for user experience
    progress_bar = st.progress(0)
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics
            data.append({
                'Ticker': ticker,
                'Name': info.get('shortName', ticker),
                'Price': info.get('currentPrice', 0),
                'Sector': info.get('sector', 'Unknown'),
                'Beta': info.get('beta', 1.0), # Risk metric
                'PE': info.get('trailingPE', 0),
                'DivYield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
                'MarketCap': info.get('marketCap', 0),
                'Description': info.get('longBusinessSummary', 'No description available.')
            })
        except Exception as e:
            pass # Skip ticker if data fails
        
        # Update progress
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(data)

def filter_stocks(df, risk_profile):
    """Filters stocks based on user risk tolerance."""
    
    # 1. Sort by Market Cap (Stability proxy) as a baseline
    df = df.sort_values(by='MarketCap', ascending=False)
    
    if risk_profile == 'Low':
        # Focus on low volatility (Beta < 1.0) and dividends
        filtered = df[ (df['Beta'] < 1.1) & (df['DivYield'] > 0.015) ]
        return filtered.head(10)
    
    elif risk_profile == 'Medium':
        # Balanced mix: Some growth (Beta < 1.4), some dividends
        filtered = df[df['Beta'] < 1.4]
        return filtered.head(10)
    
    else: # High
        # Focus on Growth, allow higher volatility
        return df.head(10)

def generate_swot(row):
    """Simulates an AI SWOT analysis based on data fields."""
    # This replaces the LLM generation for the standalone app
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
    # Logic-based SWOT
    if row['MarketCap'] > 200_000_000_000:
        strengths.append("Dominant market leader with massive scale.")
    if row['DivYield'] > 0.02:
        strengths.append(f"Strong dividend payer ({row['DivYield']*100:.2f}% yield).")
    
    if row['PE'] > 40:
        weaknesses.append("High valuation (Expensive P/E ratio).")
    elif row['PE'] < 15 and row['PE'] > 0:
        strengths.append("Undervalued compared to tech peers.")
        
    if row['Sector'] == 'Technology':
        opportunities.append("Exposure to AI and cloud computing growth.")
        threats.append("Regulatory scrutiny and rapid innovation cycles.")
    elif row['Sector'] == 'Consumer Defensive':
        opportunities.append("Steady demand regardless of recession.")
        threats.append("Slow growth during economic booms.")
    elif row['Sector'] == 'Financial Services':
        opportunities.append("Benefits from higher interest rate environments.")
        threats.append("Sensitive to economic downturns.")
        
    if row['Beta'] > 1.3:
        threats.append("High volatility stock.")
        
    return {
        "Strengths": strengths if strengths else ["Strong brand recognition."],
        "Weaknesses": weaknesses if weaknesses else ["Competitive market pressure."],
        "Opportunities": opportunities if opportunities else ["Global expansion potential."],
        "Threats": threats if threats else ["Economic headwinds."]
    }

def calculate_allocation(df, total_budget, risk_profile):
    """Allocates budget based on risk profile."""
    # Simple weighted allocation logic
    allocations = []
    remaining_budget = total_budget
    
    # We will distribute based on a 'Safety Score' derived from Beta
    # Lower beta = Higher allocation for Low Risk
    
    if risk_profile == 'Low':
        # Inverse weight to Beta (Lower beta gets more money)
        df['Weight_Score'] = 1 / (df['Beta'] + 0.1) 
    elif risk_profile == 'High':
        # Direct weight to Beta (Higher volatility gets more money)
        df['Weight_Score'] = df['Beta']
    else:
        # Equal weight roughly
        df['Weight_Score'] = 1
        
    total_score = df['Weight_Score'].sum()
    df['Allocation_Amt'] = (df['Weight_Score'] / total_score) * total_budget
    
    # Rounding
    df['Allocation_Amt'] = df['Allocation_Amt'].round(2)
    df['Shares'] = (df['Allocation_Amt'] / df['Price']).round(4)
    
    return df

# --- Sidebar: User Profile ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("User Profile")
    
    with st.form("profile_form"):
        job = st.text_input("Job/Occupation", value="Student")
        budget = st.number_input("Total Budget ($)", min_value=100, value=5000, step=100)
        risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=0)
        goal = st.selectbox("Investment Goal", ["Long-term Growth", "Short-term Gain", "Retirement", "Passive Income"])
        knowledge = st.select_slider("Knowledge Level", options=["Beginner", "Intermediate", "Expert"])
        market = st.selectbox("Target Market", ["US Stocks (S&P 500)", "Global (Simulated)"])
        
        submit_btn = st.form_submit_button("Generate Strategy")

# --- Main Content ---
st.markdown('<div class="main-header">WealthGenie Financial Advisor</div>', unsafe_allow_html=True)
st.markdown(f"**Current Strategy for:** {job} | **Budget:** ${budget:,.2f} | **Risk:** {risk}")

if submit_btn:
    # --- Step 2: Search & Filter ---
    st.markdown('<div class="sub-header">Step 1 & 2: Market Scan & Filtering</div>', unsafe_allow_html=True)
    st.write(f"Scanning market leaders for **{risk}** risk profile...")
    
    # Fetch Data
    raw_df = fetch_stock_data(STOCK_UNIVERSE)
    
    # Filter Data
    top_10_df = filter_stocks(raw_df, risk)
    
    st.success(f"Identified {len(top_10_df)} companies matching your {risk} risk profile.")
    
    # --- Step 3: Analysis (SWOT) ---
    st.markdown('<div class="sub-header">Step 3: AI Analysis (SWOT)</div>', unsafe_allow_html=True)
    
    for index, row in top_10_df.iterrows():
        swot = generate_swot(row)
        
        with st.expander(f"📊 {row['Ticker']} - {row['Name']} (${row['Price']})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Strengths:**")
                for s in swot['Strengths']: st.markdown(f"- {s}")
                st.markdown(f"**Weaknesses:**")
                for w in swot['Weaknesses']: st.markdown(f"- {w}")
            with c2:
                st.markdown(f"**Opportunities:**")
                for o in swot['Opportunities']: st.markdown(f"- {o}")
                st.markdown(f"**Threats:**")
                for t in swot['Threats']: st.markdown(f"- {t}")
            
            st.caption(f"Sector: {row['Sector']} | Beta: {row['Beta']} | Div Yield: {row['DivYield']:.2%}")

    # --- Step 4: Allocation ---
    st.markdown('<div class="sub-header">Step 4: Your Investment Strategy</div>', unsafe_allow_html=True)
    
    allocated_df = calculate_allocation(top_10_df, budget, risk)
    
    # Display Allocation Metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Allocation Breakdown")
        fig = px.pie(allocated_df, values='Allocation_Amt', names='Ticker', hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Buy Order Checklist")
        st.write("Execute these trades in your brokerage app (e.g., Fidelity, Schwab, Robinhood).")
        
        display_table = allocated_df[['Ticker', 'Name', 'Price', 'Shares', 'Allocation_Amt']].copy()
        display_table.columns = ['Ticker', 'Company', 'Price ($)', 'Fractional Shares', 'Invest ($)']
        st.dataframe(display_table, hide_index=True)
        
        st.info(f"**Note:** Ensure your brokerage supports fractional shares. If not, round the 'Shares' to the nearest whole number based on the 'Invest ($)' amount.")

else:
    # --- Landing Page State ---
    st.info("👈 Please enter your details in the sidebar and click 'Generate Strategy' to begin.")
    
    st.markdown("""
    ### How it works:
    1. **Profile Analysis:** We analyze your budget and risk tolerance.
    2. **Real-Time Scanning:** We fetch live data from the top 50 US companies.
    3. **Intelligent Filtering:** We select the 10 best stocks for YOU.
    4. **SWOT Analysis:** We evaluate the Strengths and Weaknesses of each pick.
    5. **Allocation Plan:** We tell you exactly how much to buy.
    """)