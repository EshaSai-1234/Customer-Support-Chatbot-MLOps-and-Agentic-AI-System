"""
Custom CSS and Styling Components for Streamlit Dashboard
"""

def get_custom_css() -> str:
    return """
    <style>
    /* Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .delta-green { color: #10b981; }
    .delta-amber { color: #f59e0b; }
    .delta-red { color: #ef4444; }

    /* Badge Pills */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-critical { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-high { background-color: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
    .badge-medium { background-color: #fef9c3; color: #854d0e; border: 1px solid #facc15; }
    .badge-low { background-color: #dcfce7; color: #166534; border: 1px solid #4ade80; }
    .badge-vip { background-color: #ede9fe; color: #5b21b6; border: 1px solid #a78bfa; }

    /* Trace Step Box */
    .trace-box {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
    }
    .trace-title {
        font-weight: 600;
        color: #60a5fa;
    }
    .trace-meta {
        color: #94a3b8;
        font-size: 0.8rem;
    }

    /* Recommendation Callout */
    .recommendation-card {
        background: #064e3b;
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 16px;
        color: #ecfdf5;
        margin-top: 12px;
    }
    </style>
    """
