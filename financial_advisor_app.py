import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

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
    progress_bar = st.progress(0)
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price with fallback options
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if current_price == 0 or current_price is None:
                # Try to get from history
                hist = stock.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    current_price = 0
            
            # Only add stock if we have a valid price and market cap
            market_cap = info.get('marketCap', 0)
            if current_price > 0 and market_cap > 0:
                data.append({
                    'Ticker': ticker,
                    'Name': info.get('shortName', info.get('longName', ticker)),
                    'Price': float(current_price),
                    'Sector': info.get('sector', 'Unknown'),
                    'Beta': float(info.get('beta', 1.0)) if info.get('beta') else 1.0,
                    'PE': float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
                    'DivYield': float(info.get('dividendYield', 0)) if info.get('dividendYield') else 0,
                    'MarketCap': int(market_cap),
                    'Description': info.get('longBusinessSummary', 'No description available.')
                })
        except Exception as e:
            # Skip ticker if data fails
            continue
        
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    df = pd.DataFrame(data)
    
    # Ensure all required columns exist
    required_cols = ['Ticker', 'Name', 'Price', 'Sector', 'Beta', 'PE', 'DivYield', 'MarketCap', 'Description']
    for col in required_cols:
        if col not in df.columns:
            if col == 'MarketCap':
                df[col] = 0
            elif col in ['Price', 'Beta', 'PE', 'DivYield']:
                df[col] = 0.0
            else:
                df[col] = ''
    
    return df

def filter_stocks(df, risk_profile):
    """Filters stocks based on user risk tolerance."""
    # Remove any rows with missing critical data
    df = df[df['MarketCap'] > 0].copy()
    df = df[df['Price'] > 0].copy()
    
    if len(df) == 0:
        st.error("Unable to fetch sufficient stock data. Please try again.")
        return pd.DataFrame()
    
    df = df.sort_values(by='MarketCap', ascending=False)
    
    if risk_profile == 'Low':
        filtered = df[(df['Beta'] < 1.1) & (df['DivYield'] > 0.015)]
        # If not enough stocks match criteria, relax constraints
        if len(filtered) < 5:
            filtered = df[df['Beta'] < 1.2]
        if len(filtered) < 5:
            filtered = df
        return filtered.head(10)
    elif risk_profile == 'Medium':
        filtered = df[df['Beta'] < 1.4]
        if len(filtered) < 5:
            filtered = df
        return filtered.head(10)
    else:
        return df.head(10)

def generate_swot(row):
    """Simulates an AI SWOT analysis based on data fields."""
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
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

def select_optimal_stocks(df, risk_profile, budget):
    """Select optimal subset of stocks based on risk profile and diversification."""
    # Sort by a composite score
    df = df.copy()
    
    if risk_profile == 'Low':
        # Prioritize low beta and dividends
        df['Score'] = (1 / (df['Beta'] + 0.1)) * 100 + (df['DivYield'] * 1000)
        num_stocks = min(5, len(df))  # 5 stocks for diversification
        strategy_reason = "Low risk investors benefit from fewer, more stable holdings with dividend income."
    elif risk_profile == 'Medium':
        # Balance between stability and growth
        df['Score'] = (df['MarketCap'] / 1e9) * (1 / (df['Beta'] + 0.1))
        num_stocks = min(6, len(df))  # 6 stocks for balanced portfolio
        strategy_reason = "Moderate risk portfolios balance growth potential with stability through diversification."
    else:  # High
        # Focus on growth potential
        df['Score'] = df['MarketCap'] / 1e9
        num_stocks = min(7, len(df))  # 7 stocks for aggressive growth
        strategy_reason = "Aggressive portfolios maximize growth through broader diversification across market leaders."
    
    # Ensure sector diversification
    df = df.sort_values('Score', ascending=False)
    selected = []
    sectors_used = set()
    
    # First pass: one per sector
    for _, row in df.iterrows():
        if row['Sector'] not in sectors_used and len(selected) < num_stocks:
            selected.append(row)
            sectors_used.add(row['Sector'])
    
    # Second pass: fill remaining slots with top scores
    if len(selected) < num_stocks:
        for _, row in df.iterrows():
            if len(selected) >= num_stocks:
                break
            if not any(s['Ticker'] == row['Ticker'] for s in selected):
                selected.append(row)
    
    result_df = pd.DataFrame(selected)
    return result_df, strategy_reason

def calculate_allocation(df, total_budget, risk_profile):
    """Allocates budget based on risk profile."""
    allocations = []
    
    if risk_profile == 'Low':
        df['Weight_Score'] = 1 / (df['Beta'] + 0.1)
        allocation_logic = "Lower volatility stocks receive larger allocations to minimize risk exposure."
    elif risk_profile == 'High':
        df['Weight_Score'] = df['Beta']
        allocation_logic = "Higher volatility stocks receive larger allocations to maximize growth potential."
    else:
        df['Weight_Score'] = 1
        allocation_logic = "Equal weighting balances risk and reward across all holdings."
        
    total_score = df['Weight_Score'].sum()
    df['Allocation_Amt'] = (df['Weight_Score'] / total_score) * total_budget
    df['Allocation_Amt'] = df['Allocation_Amt'].round(2)
    df['Shares'] = (df['Allocation_Amt'] / df['Price']).round(4)
    
    return df, allocation_logic

def generate_pdf_report(user_profile, allocated_df, swot_data):
    """Generate a PDF report of the investment strategy."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E88E5'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#424242'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("WealthGenie Investment Strategy Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # User Profile Section
    story.append(Paragraph("User Profile", heading_style))
    profile_data = [
        ['Job/Occupation:', user_profile['job']],
        ['Total Budget:', f"${user_profile['budget']:,.2f}"],
        ['Risk Tolerance:', user_profile['risk']],
        ['Investment Goal:', user_profile['goal']],
        ['Knowledge Level:', user_profile['knowledge']],
        ['Target Market:', user_profile['market']]
    ]
    profile_table = Table(profile_data, colWidths=[2*inch, 4*inch])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Investment Allocation
    story.append(Paragraph("Recommended Investment Allocation", heading_style))
    
    alloc_data = [['Ticker', 'Company', 'Price', 'Shares', 'Investment', 'Allocation %']]
    total_budget = user_profile['budget']
    
    for _, row in allocated_df.iterrows():
        alloc_pct = (row['Allocation_Amt'] / total_budget) * 100
        alloc_data.append([
            row['Ticker'],
            row['Name'][:25] + '...' if len(row['Name']) > 25 else row['Name'],
            f"${row['Price']:.2f}",
            f"{row['Shares']:.4f}",
            f"${row['Allocation_Amt']:.2f}",
            f"{alloc_pct:.1f}%"
        ])
    
    alloc_table = Table(alloc_data, colWidths=[0.8*inch, 2*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
    alloc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E88E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6)
    ]))
    story.append(alloc_table)
    story.append(Spacer(1, 0.3*inch))
    
    # SWOT Analysis for each stock
    story.append(PageBreak())
    story.append(Paragraph("Detailed SWOT Analysis", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    for ticker, swot in swot_data.items():
        stock_info = allocated_df[allocated_df['Ticker'] == ticker].iloc[0]
        
        story.append(Paragraph(f"<b>{ticker} - {stock_info['Name']}</b>", styles['Heading3']))
        story.append(Paragraph(f"<i>Sector: {stock_info['Sector']} | Price: ${stock_info['Price']:.2f} | Beta: {stock_info['Beta']:.2f}</i>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # SWOT details
        swot_sections = [
            ('<b>Strengths:</b>', swot['Strengths']),
            ('<b>Weaknesses:</b>', swot['Weaknesses']),
            ('<b>Opportunities:</b>', swot['Opportunities']),
            ('<b>Threats:</b>', swot['Threats'])
        ]
        
        for section_title, items in swot_sections:
            story.append(Paragraph(section_title, styles['Normal']))
            for item in items:
                story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This report is for informational purposes only and does not constitute financial advice. 
    Past performance does not guarantee future results. All investments carry risk, including 
    the potential loss of principal. Please consult with a licensed financial advisor before 
    making any investment decisions. WealthGenie is a demonstration tool and should not be 
    used as the sole basis for investment decisions.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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
    # Store data in session state for PDF generation
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # --- Step 1 & 2: Market Scan & Filtering ---
    st.markdown('<div class="sub-header">Step 1 & 2: Market Scan & Filtering</div>', unsafe_allow_html=True)
    st.write(f"Scanning market leaders for **{risk}** risk profile...")
    
    raw_df = fetch_stock_data(STOCK_UNIVERSE)
    
    # Check if we got enough data
    if len(raw_df) < 5:
        st.error("⚠️ Unable to fetch sufficient stock data. Please try again in a few moments.")
        st.stop()
    
    top_10_df = filter_stocks(raw_df, risk)
    
    if len(top_10_df) == 0:
        st.error("⚠️ No stocks found matching criteria. Please try again.")
        st.stop()
    
    st.success(f"Identified {len(top_10_df)} companies matching your {risk} risk profile.")
    
    # --- Step 3: Analysis (SWOT) ---
    st.markdown('<div class="sub-header">Step 3: AI Analysis (SWOT)</div>', unsafe_allow_html=True)
    
    swot_data = {}
    for index, row in top_10_df.iterrows():
        swot = generate_swot(row)
        swot_data[row['Ticker']] = swot
        
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

    # --- Step 4: Optimal Selection & Allocation ---
    st.markdown('<div class="sub-header">Step 4: Your Investment Strategy</div>', unsafe_allow_html=True)
    
    optimal_stocks, strategy_reason = select_optimal_stocks(top_10_df, risk, budget)
    allocated_df, allocation_logic = calculate_allocation(optimal_stocks, budget, risk)
    
    # Generate personalized explanation
    st.markdown("### 📋 Strategy Explanation")
    
    explanation_col1, explanation_col2 = st.columns([1, 1])
    
    with explanation_col1:
        st.markdown("#### Why These Stocks?")
        st.write(f"**Selection Criteria for {risk} Risk Profile:**")
        
        if risk == 'Low':
            st.markdown("""
            - ✅ **Low Volatility (Beta < 1.1):** Stocks that move less than the market
            - ✅ **Dividend Payers:** Companies that provide steady income
            - ✅ **Large Market Cap:** Established, financially stable companies
            - ✅ **Sector Diversification:** Spread across different industries to reduce risk
            """)
        elif risk == 'Medium':
            st.markdown("""
            - ✅ **Balanced Volatility (Beta < 1.4):** Moderate market movement
            - ✅ **Growth + Income Mix:** Combination of growth stocks and dividend payers
            - ✅ **Strong Market Position:** Well-established companies with growth potential
            - ✅ **Sector Diversification:** Balanced exposure across multiple sectors
            """)
        else:  # High
            st.markdown("""
            - ✅ **Growth Focus:** Market leaders with high growth potential
            - ✅ **Higher Volatility Accepted:** Willing to accept market swings for returns
            - ✅ **Innovation Leaders:** Companies at the forefront of their industries
            - ✅ **Broad Diversification:** More stocks to capture various opportunities
            """)
        
        st.info(f"💡 **Portfolio Size:** {strategy_reason}")
    
    with explanation_col2:
        st.markdown("#### Why These Percentages?")
        st.write(f"**Allocation Method for {risk} Risk:**")
        st.markdown(f"_{allocation_logic}_")
        
        # Show specific allocation reasoning
        st.markdown("**Your Allocations:**")
        for _, row in allocated_df.iterrows():
            pct = (row['Allocation_Amt'] / budget) * 100
            
            reason = ""
            if risk == 'Low':
                if row['Beta'] < 0.9:
                    reason = f"(Very stable, Beta: {row['Beta']:.2f})"
                elif row['DivYield'] > 0.025:
                    reason = f"(Strong dividend: {row['DivYield']*100:.2f}%)"
                else:
                    reason = f"(Stable, Beta: {row['Beta']:.2f})"
            elif risk == 'Medium':
                if row['MarketCap'] > 500_000_000_000:
                    reason = f"(Market leader: ${row['MarketCap']/1e9:.0f}B)"
                else:
                    reason = f"(Growth potential in {row['Sector']})"
            else:  # High
                if row['Beta'] > 1.2:
                    reason = f"(High growth, Beta: {row['Beta']:.2f})"
                else:
                    reason = f"(Market leader: ${row['MarketCap']/1e9:.0f}B)"
            
            st.markdown(f"- **{row['Ticker']}:** {pct:.1f}% {reason}")
    
    st.markdown("---")
    
    st.info(f"✨ **Final Strategy:** Based on your **{job}** profile with a **${budget:,}** budget seeking **{goal.lower()}**, we've selected **{len(allocated_df)}** stocks out of 10 candidates for optimal diversification and risk management.")
    
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
    
    # Store data for PDF generation
    st.session_state.analysis_complete = True
    st.session_state.user_profile = {
        'job': job,
        'budget': budget,
        'risk': risk,
        'goal': goal,
        'knowledge': knowledge,
        'market': market
    }
    st.session_state.allocated_df = allocated_df
    st.session_state.swot_data = {k: v for k, v in swot_data.items() if k in allocated_df['Ticker'].values}
    
    # --- PDF Generation Button ---
    st.markdown('<div class="sub-header">Export Your Strategy</div>', unsafe_allow_html=True)
    
    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating your personalized PDF report..."):
            pdf_buffer = generate_pdf_report(
                st.session_state.user_profile,
                st.session_state.allocated_df,
                st.session_state.swot_data
            )
            
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name=f"WealthGenie_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF Report generated successfully!")

elif st.session_state.get('analysis_complete', False):
    # Show PDF button if analysis was already done
    st.markdown('<div class="sub-header">Export Your Strategy</div>', unsafe_allow_html=True)
    
    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating your personalized PDF report..."):
            pdf_buffer = generate_pdf_report(
                st.session_state.user_profile,
                st.session_state.allocated_df,
                st.session_state.swot_data
            )
            
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name=f"WealthGenie_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF Report generated successfully!")

else:
    # --- Landing Page State ---
    st.info("👈 Please enter your details in the sidebar and click 'Generate Strategy' to begin.")
    
    st.markdown("""
    ### How it works:
    1. **Profile Analysis:** We analyze your budget and risk tolerance.
    2. **Real-Time Scanning:** We fetch live data from the top 50 US companies.
    3. **Intelligent Filtering:** We select the 10 best stocks for YOU.
    4. **SWOT Analysis:** We evaluate the Strengths and Weaknesses of each pick.
    5. **Optimal Selection:** We choose the best subset for your portfolio.
    6. **Allocation Plan:** We tell you exactly how much to buy.
    7. **PDF Export:** Download your complete investment strategy report.
    """)
