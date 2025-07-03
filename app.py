import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Leadership Programme ROI Calculator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #666;
        font-weight: 500;
    }
    .metric-status {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client (optional)
@st.cache_resource
def init_groq():
    try:
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            return Groq(api_key=api_key)
    except:
        pass
    return None

groq_client = init_groq()

# Industry Templates
INDUSTRY_TEMPLATES = {
    'technology': {
        'name': "Technology",
        'avg_salary': 110000,
        'current_turnover': 22,
        'replacement_cost': 2.0,
        'productivity_gain': 18,
        'retention_improvement': 30,
        'team_performance_gain': 15
    },
    'finance': {
        'name': "Financial Services",
        'avg_salary': 105000,
        'current_turnover': 15,
        'replacement_cost': 1.8,
        'productivity_gain': 12,
        'retention_improvement': 20,
        'team_performance_gain': 10
    },
    'healthcare': {
        'name': "Healthcare",
        'avg_salary': 85000,
        'current_turnover': 20,
        'replacement_cost': 1.6,
        'productivity_gain': 14,
        'retention_improvement': 25,
        'team_performance_gain': 12
    },
    'manufacturing': {
        'name': "Manufacturing",
        'avg_salary': 80000,
        'current_turnover': 16,
        'replacement_cost': 1.4,
        'productivity_gain': 16,
        'retention_improvement': 22,
        'team_performance_gain': 14
    },
    'consulting': {
        'name': "Consulting",
        'avg_salary': 120000,
        'current_turnover': 25,
        'replacement_cost': 2.2,
        'productivity_gain': 20,
        'retention_improvement': 35,
        'team_performance_gain': 18
    }
}

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.0f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%"

def get_roi_color_status(roi):
    """Get color and status for ROI"""
    if roi >= 300:
        return "🟢 Excellent", "#10B981"
    elif roi >= 200:
        return "🟡 Good", "#F59E0B"
    elif roi >= 100:
        return "🟠 Moderate", "#EF4444"
    else:
        return "🔴 Review Required", "#DC2626"

def get_payback_color_status(months):
    """Get color and status for payback period"""
    if months <= 12:
        return "🟢 Fast", "#10B981"
    elif months <= 18:
        return "🟡 Moderate", "#F59E0B"
    elif months <= 24:
        return "🟠 Slow", "#EF4444"
    else:
        return "🔴 Very Slow", "#DC2626"

def calculate_roi(params):
    """Calculate ROI and related metrics"""
    # Program Costs
    participant_time_cost = (
        params['participants'] * 
        (params['avg_salary'] * 1.3 / 12) * 
        (params['time_commitment'] / 160) * 
        params['program_duration']
    )
    
    total_program_costs = (
        params['facilitator_costs'] + params['materials_costs'] + 
        params['venue_costs'] + params['travel_costs'] + 
        params['technology_costs'] + params['assessment_costs'] + 
        participant_time_cost
    )
    
    # Annual Benefits
    productivity_benefit = params['participants'] * params['avg_salary'] * (params['productivity_gain'] / 100)
    
    retention_savings = (
        params['participants'] * (params['current_turnover'] / 100) * 
        (params['retention_improvement'] / 100) * params['avg_salary'] * params['replacement_cost']
    )
    
    team_productivity_benefit = (
        params['participants'] * params['team_size'] * 
        (params['avg_salary'] * 0.7) * (params['team_performance_gain'] / 100)
    )
    
    promotion_benefit = (
        params['participants'] * 0.3 * (params['promotion_acceleration'] / 12) * 
        (params['avg_salary'] * 0.2)
    )
    
    decision_benefit = (
        params['participants'] * params['avg_salary'] * 0.1 * 
        (params['decision_quality_gain'] / 100)
    )
    
    total_annual_benefits = (
        productivity_benefit + retention_savings + team_productivity_benefit + 
        promotion_benefit + decision_benefit
    )
    
    # Multi-year analysis
    total_benefits = total_annual_benefits * params['analysis_years']
    net_benefit = total_benefits - total_program_costs
    roi = (net_benefit / total_program_costs) * 100 if total_program_costs > 0 else 0
    payback_months = (total_program_costs / (total_annual_benefits / 12)) if total_annual_benefits > 0 else float('inf')
    
    # NPV calculation (8% discount rate)
    discount_rate = 0.08
    npv = -total_program_costs
    for year in range(1, params['analysis_years'] + 1):
        npv += total_annual_benefits / ((1 + discount_rate) ** year)
    
    benefit_cost_ratio = total_benefits / total_program_costs if total_program_costs > 0 else 0
    
    return {
        'costs': {
            'facilitator': params['facilitator_costs'],
            'materials': params['materials_costs'],
            'venue': params['venue_costs'],
            'travel': params['travel_costs'],
            'technology': params['technology_costs'],
            'assessment': params['assessment_costs'],
            'participant_time': participant_time_cost,
            'total': total_program_costs
        },
        'benefits': {
            'productivity': productivity_benefit,
            'retention': retention_savings,
            'team_performance': team_productivity_benefit,
            'promotion': promotion_benefit,
            'decision_quality': decision_benefit,
            'total_annual': total_annual_benefits,
            'total_multi_year': total_benefits
        },
        'kpis': {
            'roi': roi,
            'payback_months': payback_months,
            'npv': npv,
            'net_benefit': net_benefit,
            'benefit_cost_ratio': benefit_cost_ratio
        }
    }

def get_ai_insights(results, params):
    """Get AI-powered insights using Groq"""
    if not groq_client:
        return "AI insights unavailable (Groq API key not configured)"
    
    try:
        prompt = f"""
        Analyze this leadership development program ROI calculation and provide insights:
        
        Program Details:
        - Participants: {params['participants']}
        - Duration: {params['program_duration']} months
        - Investment: {format_currency(results['costs']['total'])}
        
        Key Metrics:
        - ROI: {results['kpis']['roi']:.1f}%
        - Payback: {results['kpis']['payback_months']:.1f} months
        - NPV: {format_currency(results['kpis']['npv'])}
        
        Provide 3-4 bullet points with actionable insights and recommendations.
        Focus on business value, risk mitigation, and optimization opportunities.
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🎯 Leadership Programme ROI Calculator</h1>
        <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
            Calculate ROI, payback period, and build compelling business cases for leadership development initiatives
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'params' not in st.session_state:
        st.session_state.params = {
            # Program Parameters
            'participants': 20,
            'program_duration': 6,
            'avg_salary': 95000,
            'time_commitment': 15,
            'analysis_years': 3,
            
            # Program Costs
            'facilitator_costs': 75000,
            'materials_costs': 15000,
            'venue_costs': 25000,
            'travel_costs': 30000,
            'technology_costs': 12000,
            'assessment_costs': 8000,
            
            # Benefit Assumptions
            'productivity_gain': 15,
            'retention_improvement': 25,
            'promotion_acceleration': 6,
            'team_performance_gain': 12,
            'decision_quality_gain': 20,
            
            # Industry Benchmarks
            'current_turnover': 18,
            'replacement_cost': 1.5,
            'team_size': 8
        }
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Program Configuration")
        
        # Industry Templates
        st.subheader("🏭 Industry Templates")
        industry_options = [''] + [template['name'] for template in INDUSTRY_TEMPLATES.values()]
        selected_industry = st.selectbox("Select Industry Template", industry_options)
        
        if selected_industry:
            # Find the template key
            template_key = None
            for key, template in INDUSTRY_TEMPLATES.items():
                if template['name'] == selected_industry:
                    template_key = key
                    break
            
            if template_key and st.button("Apply Template"):
                template = INDUSTRY_TEMPLATES[template_key]
                st.session_state.params.update({
                    'avg_salary': template['avg_salary'],
                    'current_turnover': template['current_turnover'],
                    'replacement_cost': template['replacement_cost'],
                    'productivity_gain': template['productivity_gain'],
                    'retention_improvement': template['retention_improvement'],
                    'team_performance_gain': template['team_performance_gain']
                })
                st.success(f"Applied {selected_industry} template!")
        
        st.divider()
        
        # Program Parameters
        st.subheader("📈 Program Parameters")
        st.session_state.params['participants'] = st.number_input(
            "Participants", min_value=1, value=st.session_state.params['participants']
        )
        st.session_state.params['program_duration'] = st.number_input(
            "Duration (months)", min_value=1, value=st.session_state.params['program_duration']
        )
        st.session_state.params['avg_salary'] = st.number_input(
            "Average Salary ($)", min_value=0, value=st.session_state.params['avg_salary'], step=5000
        )
        st.session_state.params['time_commitment'] = st.number_input(
            "Time Commitment (hours/month)", min_value=1, value=st.session_state.params['time_commitment']
        )
        st.session_state.params['analysis_years'] = st.number_input(
            "Analysis Period (years)", min_value=1, max_value=10, value=st.session_state.params['analysis_years']
        )
        
        st.divider()
        
        # Program Costs
        st.subheader("💰 Program Costs")
        cost_fields = [
            ('facilitator_costs', 'Facilitator Costs ($)', 5000),
            ('materials_costs', 'Materials & Content ($)', 1000),
            ('venue_costs', 'Venue & Catering ($)', 1000),
            ('travel_costs', 'Travel & Accommodation ($)', 1000),
            ('technology_costs', 'Technology Platform ($)', 1000),
            ('assessment_costs', 'Assessment & Evaluation ($)', 1000)
        ]
        
        for field, label, step in cost_fields:
            st.session_state.params[field] = st.number_input(
                label, min_value=0, value=st.session_state.params[field], step=step
            )
    
    # Calculate results
    results = calculate_roi(st.session_state.params)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "⚙️ Assumptions", "📈 Analysis", "🤖 AI Insights"])
    
    with tab1:
        # Key Metrics Dashboard
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            roi_status, roi_color = get_roi_color_status(results['kpis']['roi'])
            st.metric(
                "Return on Investment",
                f"{results['kpis']['roi']:.0f}%",
                delta=roi_status
            )
        
        with col2:
            payback_status, payback_color = get_payback_color_status(results['kpis']['payback_months'])
            st.metric(
                "Payback Period",
                f"{results['kpis']['payback_months']:.1f} months",
                delta=payback_status
            )
        
        with col3:
            npv_status = "🟢 Positive" if results['kpis']['npv'] > 0 else "🔴 Negative"
            st.metric(
                "Net Present Value",
                format_currency(results['kpis']['npv']),
                delta=npv_status
            )
        
        with col4:
            bcr = results['kpis']['benefit_cost_ratio']
            bcr_status = "🟢 Strong" if bcr >= 3 else "🟡 Moderate" if bcr >= 2 else "🔴 Weak"
            st.metric(
                "Benefit-Cost Ratio",
                f"{bcr:.1f}:1",
                delta=bcr_status
            )
        
        st.divider()
        
        # Cost and Benefit Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Cost Breakdown")
            cost_data = {
                'Category': ['Facilitator', 'Materials', 'Venue', 'Travel', 'Technology', 'Assessment', 'Participant Time'],
                'Amount': [
                    results['costs']['facilitator'],
                    results['costs']['materials'],
                    results['costs']['venue'],
                    results['costs']['travel'],
                    results['costs']['technology'],
                    results['costs']['assessment'],
                    results['costs']['participant_time']
                ]
            }
            
            fig_costs = px.pie(
                values=cost_data['Amount'],
                names=cost_data['Category'],
                title=f"Total Investment: {format_currency(results['costs']['total'])}"
            )
            st.plotly_chart(fig_costs, use_container_width=True)
        
        with col2:
            st.subheader("📈 Benefit Breakdown")
            benefit_data = {
                'Category': ['Productivity', 'Retention', 'Team Performance', 'Promotions', 'Decision Quality'],
                'Annual Amount': [
                    results['benefits']['productivity'],
                    results['benefits']['retention'],
                    results['benefits']['team_performance'],
                    results['benefits']['promotion'],
                    results['benefits']['decision_quality']
                ]
            }
            
            fig_benefits = px.bar(
                x=benefit_data['Category'],
                y=benefit_data['Annual Amount'],
                title=f"Annual Benefits: {format_currency(results['benefits']['total_annual'])}"
            )
            fig_benefits.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_benefits, use_container_width=True)
    
    with tab2:
        st.subheader("⚙️ Impact Assumptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.params['productivity_gain'] = st.slider(
                "Productivity Improvement (%)", 0, 50, st.session_state.params['productivity_gain'],
                help="Industry average: 10-20%"
            )
            st.session_state.params['retention_improvement'] = st.slider(
                "Retention Improvement (%)", 0, 50, st.session_state.params['retention_improvement'],
                help="Industry average: 15-30%"
            )
            st.session_state.params['team_performance_gain'] = st.slider(
                "Team Performance Gain (%)", 0, 30, st.session_state.params['team_performance_gain'],
                help="Industry average: 8-15%"
            )
        
        with col2:
            st.session_state.params['decision_quality_gain'] = st.slider(
                "Decision Quality Improvement (%)", 0, 40, st.session_state.params['decision_quality_gain'],
                help="Industry average: 15-25%"
            )
            st.session_state.params['current_turnover'] = st.slider(
                "Current Turnover Rate (%)", 0, 50, st.session_state.params['current_turnover'],
                help="Annual turnover rate"
            )
            st.session_state.params['replacement_cost'] = st.slider(
                "Replacement Cost Multiple", 0.5, 3.0, st.session_state.params['replacement_cost'], 0.1,
                help="Multiple of salary to replace employee"
            )
    
    with tab3:
        st.subheader("📊 Scenario Analysis")
        
        # Scenario comparison
        scenarios = {
            'Conservative (-25%)': 0.75,
            'Realistic (Current)': 1.0,
            'Optimistic (+25%)': 1.25
        }
        
        scenario_data = []
        for scenario_name, multiplier in scenarios.items():
            scenario_roi = results['kpis']['roi'] * multiplier
            scenario_payback = results['kpis']['payback_months'] / multiplier
            scenario_net_benefit = results['kpis']['net_benefit'] * multiplier
            
            scenario_data.append({
                'Scenario': scenario_name,
                'ROI (%)': f"{scenario_roi:.0f}%",
                'Payback (months)': f"{scenario_payback:.1f}",
                'Net Benefit': format_currency(scenario_net_benefit)
            })
        
        df_scenarios = pd.DataFrame(scenario_data)
        st.table(df_scenarios)
        
        st.divider()
        
        # Cash flow visualization
        st.subheader("💰 Cash Flow Analysis")
        
        years = list(range(0, st.session_state.params['analysis_years'] + 1))
        cumulative_cash_flow = [-results['costs']['total']]
        
        for year in range(1, st.session_state.params['analysis_years'] + 1):
            cumulative_cash_flow.append(
                cumulative_cash_flow[-1] + results['benefits']['total_annual']
            )
        
        fig_cashflow = go.Figure()
        fig_cashflow.add_trace(go.Scatter(
            x=years,
            y=cumulative_cash_flow,
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(width=3)
        ))
        fig_cashflow.add_hline(y=0, line_dash="dash", line_color="red", 
                              annotation_text="Break-even")
        fig_cashflow.update_layout(
            title="Cumulative Cash Flow Over Time",
            xaxis_title="Year",
            yaxis_title="Cumulative Cash Flow ($)",
            hovermode='x'
        )
        st.plotly_chart(fig_cashflow, use_container_width=True)
    
    with tab4:
        st.subheader("🤖 AI-Powered Insights")
        
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing your business case..."):
                insights = get_ai_insights(results, st.session_state.params)
                st.markdown(insights)
        
        st.divider()
        
        st.subheader("💡 Best Practices & Recommendations")
        
        recommendations = [
            "**Measurement:** Implement Kirkpatrick Level 3 & 4 evaluation to track behavior change and business results",
            "**Timeline:** Benefits typically materialize 3-6 months post-program completion",
            "**Sustainability:** Include manager coaching and 90-day action plans for lasting impact",
            "**Benchmark:** World-class leadership programs typically achieve 200-400% ROI",
            "**Risk Mitigation:** Start with pilot cohort to validate assumptions before full rollout"
        ]
        
        for rec in recommendations:
            st.markdown(f"✓ {rec}")
    
    # Export functionality
    st.divider()
    st.subheader("📄 Export Business Case")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data (JSON)", type="secondary"):
            export_data = {
                'program': {k: v for k, v in st.session_state.params.items()},
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"leadership_roi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📄 Export Summary (CSV)", type="secondary"):
            summary_data = {
                'Metric': ['ROI (%)', 'Payback (months)', 'NPV ($)', 'Investment ($)', 'Annual Benefits ($)', 'Net Benefit ($)'],
                'Value': [
                    f"{results['kpis']['roi']:.1f}",
                    f"{results['kpis']['payback_months']:.1f}",
                    f"{results['kpis']['npv']:.0f}",
                    f"{results['costs']['total']:.0f}",
                    f"{results['benefits']['total_annual']:.0f}",
                    f"{results['kpis']['net_benefit']:.0f}"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"leadership_roi_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        recommendation = (
            "STRONG BUSINESS CASE - Proceed with implementation" if results['kpis']['roi'] >= 200 else
            "MODERATE BUSINESS CASE - Consider optimization" if results['kpis']['roi'] >= 100 else
            "WEAK BUSINESS CASE - Review assumptions and design"
        )
        st.success(f"**Recommendation:** {recommendation}")

if __name__ == "__main__":
    main()
