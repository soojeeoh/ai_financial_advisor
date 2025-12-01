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
    page_title="WealthGenie | Simple AI Financial Advisor",
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

def get_stability_label(beta):
    """Translates Beta into plain English."""
    if beta < 0.8: return "Very Stable (Safe)"
    if beta < 1.1: return "Stable (Average)"
    if beta < 1.4: return "Moderate Swings"
    return "High Fluctuations (Risky)"

def format_large_number(num):
    """Formats market cap to Trillions/Billions."""
    if num >= 1_000_000_000_000:
        return f"${num/1_000_000_000_000:.2f} Trillion"
    elif num >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f} Billion"
    else:
        return f"${num/1_000_000:.2f} Million"

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
    
    # Translated jargon to plain English
    if row['MarketCap'] > 200_000_000_000:
        strengths.append("One of the largest, most reliable companies in the world.")
    
    if row['DivYield'] > 0.02:
        strengths.append(f"Pays you cash just for holding the stock ({row['DivYield']*100:.2f}% yearly).")
    
    if row['PE'] > 40:
        weaknesses.append("Stock price is currently very expensive compared to its profits.")
    elif row['PE'] < 15 and row['PE'] > 0:
        strengths.append("Stock price is cheap (""on sale"") compared to other companies.")
        
    if row['Sector'] == 'Technology':
        opportunities.append("Could grow fast due to AI and computer innovation.")
        threats.append("Government rules or new inventions could hurt them.")
    elif row['Sector'] == 'Consumer Defensive':
        opportunities.append("People buy their stuff even when the economy is bad.")
        threats.append("Does not grow very fast when the economy is booming.")
    elif row['Sector'] == 'Financial Services':
        opportunities.append("Makes more money when interest rates are high.")
        threats.append("Can lose value if the economy slows down.")
        
    if row['Beta'] > 1.3:
        threats.append("The stock price is a rollercoaster—expect big ups and downs.")
    else:
        strengths.append("The stock price is usually steady and calm.")
        
    return {
        "Strengths": strengths if strengths else ["Strong brand recognition."],
        "Weaknesses": weaknesses if weaknesses else ["Lots of competition."],
        "Opportunities": opportunities if opportunities else ["Expanding to new countries."],
        "Threats": threats if threats else ["Economic slowdowns."]
    }

def select_optimal_stocks(df, risk_profile, budget):
    """Select optimal subset of stocks based on risk profile and diversification."""
    # Sort by a composite score
    df = df.copy()
    
    if risk_profile == 'Low':
        # Prioritize low beta and dividends
        df['Score'] = (1 / (df['Beta'] + 0.1)) * 100 + (df['DivYield'] * 1000)
        num_stocks = min(5, len(df))  # 5 stocks for diversification
        strategy_reason = "Conservative investors do best with a few very stable, cash-paying companies."
    elif risk_profile == 'Medium':
        # Balance between stability and growth
        df['Score'] = (df['MarketCap'] / 1e9) * (1 / (df['Beta'] + 0.1))
        num_stocks = min(6, len(df))  # 6 stocks for balanced portfolio
        strategy_reason = "Balanced portfolios mix steady giants with companies that have room to grow."
    else:  # High
        # Focus on growth potential
        df['Score'] = df['MarketCap'] / 1e9
        num_stocks = min(7, len(df))  # 7 stocks for aggressive growth
        strategy_reason = "Aggressive portfolios buy more companies to try and catch the biggest winners."

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
        allocation_logic = "We put more money into the 'safe' stocks to protect your savings."
    elif risk_profile == 'High':
        df['Weight_Score'] = df['Beta']
        allocation_logic = "We put more money into the 'wilder' stocks to try for maximum profit."
    else:
        df['Weight_Score'] = 1
        allocation_logic = "We spread the money equally so you don't rely too much on one company."
        
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
    story.append(Paragraph("WealthGenie Investment Plan", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # User Profile Section
    story.append(Paragraph("Your Profile", heading_style))
    profile_data = [
        ['Job/Occupation:', user_profile['job']],
        ['Total Budget:', f"${user_profile['budget']:,.2f}"],
        ['Risk Style:', user_profile['risk']],
        ['Goal:', user_profile['goal']],
        ['Experience:', user_profile['knowledge']],
        ['Market:', user_profile['market']]
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
    story.append(Paragraph("What You Should Buy", heading_style))
    
    alloc_data = [['Ticker', 'Company', 'Price', 'Shares', 'Invest ($)', 'Share %']]
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
    story.append(Paragraph("Why We Picked These (Analysis)", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    for ticker, swot in swot_data.items():
        stock_info = allocated_df[allocated_df['Ticker'] == ticker].iloc[0]
        
        # Translate Beta for PDF
        stability_desc = get_stability_label(stock_info['Beta'])
        
        story.append(Paragraph(f"<b>{ticker} - {stock_info['Name']}</b>", styles['Heading3']))
        story.append(Paragraph(f"<i>Industry: {stock_info['Sector']} | Price: ${stock_info['Price']:.2f} | Stability: {stability_desc}</i>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # SWOT details
        swot_sections = [
            ('<b>Good Stuff (Strengths):</b>', swot['Strengths']),
            ('<b>Bad Stuff (Weaknesses):</b>', swot['Weaknesses']),
            ('<b>Future Chances (Opportunities):</b>', swot['Opportunities']),
            ('<b>Risks (Threats):</b>', swot['Threats'])
        ]
        
        for section_title, items in swot_sections:
            story.append(Paragraph(section_title, styles['Normal']))
            for item in items:
                story.append(Paragraph(f"• {item}", styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
            
        story.append(Spacer(1, 0.2*inch))

    # Disclaimer
    story.append(PageBreak())
    story.append(Paragraph("Important Notice", heading_style))
    disclaimer_text = """
    This report is for learning purposes only. It is not official financial advice. 
    Stocks can go down as well as up, and you could lose money. Always talk to a 
    certified professional before investing real money. WealthGenie is a demonstration tool.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Sidebar: User Profile ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("Your Profile")
    
    with st.form("profile_form"):
        job = st.text_input("Job/Occupation", value="Student")
        budget = st.number_input("Amount to Invest ($)", min_value=100, value=5000, step=100)
        risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=0)
        goal = st.selectbox("Investment Goal", ["Grow my Money over 10+ Years", "Short-term Gain", "Retirement", "Monthly Income"])
        knowledge = st.select_slider("Knowledge Level", options=["Newbie", "Some Idea", "Pro"])
        market = st.selectbox("Where to Invest", ["US Stock Market (S&P 500)", "Global (Simulated)"])
        
        submit_btn = st.form_submit_button("Create My Plan")

# --- Main Content ---
st.markdown('<div class="main-header">WealthGenie Financial Advisor</div>', unsafe_allow_html=True)
st.markdown(f"**Current Strategy for:** {job} | **Budget:** ${budget:,.2f} | **Risk:** {risk}")

if submit_btn:
    # Store data in session state for PDF generation
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    # --- Step 1 & 2: Market Scan & Filtering ---
    st.markdown('<div class="sub-header">Step 1 & 2: Scanning the Market</div>', unsafe_allow_html=True)
    st.write(f"Looking for the best companies for a **{risk}** risk style...")
    
    raw_df = fetch_stock_data(STOCK_UNIVERSE)
    
    # Check if we got enough data
    if len(raw_df) < 5:
        st.error("⚠️ Trouble connecting to the stock market. Please try again in a moment.")
        st.stop()
        
    top_10_df = filter_stocks(raw_df, risk)
    
    if len(top_10_df) == 0:
        st.error("⚠️ No stocks found. Please try again.")
        st.stop()
        
    st.success(f"Found {len(top_10_df)} companies that match your profile.")

    # --- Step 3: Analysis (SWOT) ---
    st.markdown('<div class="sub-header">Step 3: AI Analysis (The Good & Bad)</div>', unsafe_allow_html=True)
    
    swot_data = {}
    for index, row in top_10_df.iterrows():
        swot = generate_swot(row)
        swot_data[row['Ticker']] = swot
        
        # Friendly formatting for display
        stability_text = get_stability_label(row['Beta'])
        size_text = format_large_number(row['MarketCap'])
        
        with st.expander(f"📊 {row['Ticker']} - {row['Name']} (${row['Price']})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**✅ The Good Stuff:**")
                for s in swot['Strengths']: st.markdown(f"- {s}")
                st.markdown(f"**❌ The Bad Stuff:**")
                for w in swot['Weaknesses']: st.markdown(f"- {w}")
            with c2:
                st.markdown(f"**🚀 Opportunities:**")
                for o in swot['Opportunities']: st.markdown(f"- {o}")
                st.markdown(f"**⚠️ Risks:**")
                for t in swot['Threats']: st.markdown(f"- {t}")
            
            st.markdown("---")
            st.caption(f"Industry: {row['Sector']} | Stability: {stability_text} | Yearly Payout: {row['DivYield']:.2%} | Company Size: {size_text}")

    # --- Step 4: Optimal Selection & Allocation ---
    st.markdown('<div class="sub-header">Step 4: Your Investment Plan</div>', unsafe_allow_html=True)
    
    optimal_stocks, strategy_reason = select_optimal_stocks(top_10_df, risk, budget)
    allocated_df, allocation_logic = calculate_allocation(optimal_stocks, budget, risk)
    
    # Generate personalized explanation
    st.markdown("### 📋 Why We Chose This?")
    
    explanation_col1, explanation_col2 = st.columns([1, 1])
    
    with explanation_col1:
        st.markdown("#### Why These Stocks?")
        st.write(f"**Criteria for {risk} Risk Profile:**")
        
        if risk == 'Low':
            st.markdown("""
            - ✅ **High Stability:** Stocks that don't bounce up and down wildly.
            - ✅ **Cash Payouts:** Companies that pay you dividends regularly.
            - ✅ **Massive Size:** Giant, reliable companies you know.
            - ✅ **Safety:** Spread across different industries to be safe.
            """)
        elif risk == 'Medium':
            st.markdown("""
            - ✅ **Balanced Stability:** Not too boring, not too crazy.
            - ✅ **Mix of Growth & Cash:** Some companies for profit, some for safety.
            - ✅ **Strong History:** Well-established companies.
            - ✅ **Safety:** Balanced spread across different industries.
            """)
        else:  # High
            st.markdown("""
            - ✅ **Pure Growth:** Companies that could double in size.
            - ✅ **Price Swings Okay:** We accept ups and downs to get bigger rewards.
            - ✅ **Innovators:** Companies inventing the future.
            - ✅ **Wide Net:** Buying more stocks to catch the winners.
            """)
            
        st.info(f"💡 **Strategy:** {strategy_reason}")

    with explanation_col2:
        st.markdown("#### Why These Amounts?")
        st.write(f"**How we split the money:**")
        st.markdown(f"_{allocation_logic}_")
        
        # Show specific allocation reasoning
        st.markdown("**Your Breakdown:**")
        for _, row in allocated_df.iterrows():
            pct = (row['Allocation_Amt'] / budget) * 100
            
            reason = ""
            if risk == 'Low':
                if row['Beta'] < 0.9:
                    reason = "(Very Safe)"
                elif row['DivYield'] > 0.025:
                    reason = "(Good Payout)"
                else:
                    reason = "(Reliable)"
            elif risk == 'Medium':
                if row['MarketCap'] > 500_000_000_000:
                    reason = "(Giant Company)"
                else:
                    reason = "(Growth Potential)"
            else:  # High
                if row['Beta'] > 1.2:
                    reason = "(High Growth)"
                else:
                    reason = "(Market Leader)"
            
            st.markdown(f"- **{row['Ticker']}:** {pct:.1f}% {reason}")

    st.markdown("---")
    st.info(f"✨ **Final Plan:** Based on your **{job}** profile with a **${budget:,}** budget, we selected **{len(allocated_df)}** distinct stocks to give you the best chance of success.")

    # Display Allocation Metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Visual Breakdown")
        fig = px.pie(allocated_df, values='Allocation_Amt', names='Ticker', hole=0.4)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Your Shopping List")
        st.write("Go to your trading app (like Robinhood or Schwab) and buy these:")
        
        display_table = allocated_df[['Ticker', 'Name', 'Price', 'Shares', 'Allocation_Amt']].copy()
        display_table.columns = ['Ticker', 'Company', 'Price ($)', 'Shares to Buy', 'Amount to Spend ($)']
        st.dataframe(display_table, hide_index=True)
        
        st.info(f"**Note:** If your app doesn't let you buy fractions of a share, just round the 'Shares to Buy' to the nearest whole number.")

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
    st.markdown('<div class="sub-header">Save Your Plan</div>', unsafe_allow_html=True)
    
    if st.button("📄 Download PDF Report", type="primary"):
        with st.spinner("Writing your report..."):
            pdf_buffer = generate_pdf_report(
                st.session_state.user_profile,
                st.session_state.allocated_df,
                st.session_state.swot_data
            )
            
            st.download_button(
                label="⬇️ Click here to Download PDF",
                data=pdf_buffer,
                file_name=f"WealthGenie_Plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.success("✅ Report ready! Check your downloads folder.")

elif st.session_state.get('analysis_complete', False):
    # Show PDF button if analysis was already done
    st.markdown('<div class="sub-header">Save Your Plan</div>', unsafe_allow_html=True)
    
    if st.button("📄 Download PDF Report", type="primary"):
        with st.spinner("Writing your report..."):
            pdf_buffer = generate_pdf_report(
                st.session_state.user_profile,
                st.session_state.allocated_df,
                st.session_state.swot_data
            )
            
            st.download_button(
                label="⬇️ Click here to Download PDF",
                data=pdf_buffer,
                file_name=f"WealthGenie_Plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.success("✅ Report ready! Check your downloads folder.")

else:
    # --- Landing Page State ---
    st.info("👈 Enter your info in the sidebar and click 'Create My Plan' to start.")
    
    st.markdown("""
    ### How it works:
    1. **Tell us about you:** We look at your budget and how much risk you want to take.
    2. **We scan the market:** We look at the top 50 biggest companies in the US.
    3. **We filter them:** We find the 10 best matches for YOU.
    4. **We check them:** We look for good things (Strengths) and bad things (Weaknesses).
    5. **We make a plan:** We tell you exactly what to buy and how much to spend.
    6. **Download:** You get a nice PDF to keep.
    """)
