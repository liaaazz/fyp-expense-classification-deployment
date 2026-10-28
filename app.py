import streamlit as st
import pandas as pd
import joblib
import re
import uuid
import hashlib
from datetime import date
import json
import sqlite3
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from src.feedback_matcher import predict_from_feedback
import html

# Load local environment variables if available
load_dotenv("apikey.env")

# Use Streamlit Secrets in the cloud, with local fallback
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = None

GEMINI_API_KEY = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured. "
        "Set it in apikey.env locally or Streamlit Secrets online."
    )

from PIL import Image
from pathlib import Path

from src.database import (
    initialize_database,
    DB_PATH,
    save_expense,
    get_user_expenses,
    delete_expense,
    save_monthly_budget,
    get_monthly_budget,
    save_learning_feedback,
)

from src.authentication import register_user, authenticate_user


def readable_column_names(dataframe):
    """Return a display copy with database-style column names made readable."""
    display_df = dataframe.copy()
    display_df.columns = [
        str(column).replace("_", " ").title()
        for column in display_df.columns
    ]
    return display_df


def readable_column_names(dataframe):
    """Return a display copy with database-style column names made readable."""
    display_df = dataframe.copy()
    display_df.columns = [
        str(column).replace("_", " ").title()
        for column in display_df.columns
    ]
    return display_df


def render_styled_table(df, app_theme):
    """Render a table that follows the selected app theme."""

    if df is None or df.empty:
        st.info("No data available.")
        return

    if app_theme == "Light":
        table_bg = "#FFFFFF"
        header_bg = "#F1F5F9"
        table_text = "#1E293B"
        border_color = "#CBD5E1"
    else:
        table_bg = "#0E1117"
        header_bg = "#262833"
        table_text = "#F1F5F9"
        border_color = "#374151"

    headers_html = "".join(
        f'<th style="padding:10px;text-align:left;'
        f'background:{header_bg};color:{table_text};'
        f'border-bottom:1px solid {border_color};">'
        f'{html.escape(str(col))}</th>'
        for col in df.columns
    )

    rows_html = ""

    for _, row in df.iterrows():
        cells_html = ""

        for value in row:
            if pd.isna(value):
                value = ""

            cells_html += (
                f'<td style="padding:9px 10px;'
                f'color:{table_text};'
                f'border-bottom:1px solid {border_color};">'
                f'{html.escape(str(value))}</td>'
            )

        rows_html += f"<tr>{cells_html}</tr>"

    table_html = f"""
    <div style="width:100%;overflow-x:auto;
                border:1px solid {border_color};border-radius:8px;">
        <table style="width:100%;border-collapse:collapse;
                      background:{table_bg};color:{table_text};
                      font-size:13px;">
            <thead><tr>{headers_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """

    st.html(table_html)

def update_expense_record(user_id, receipt_id, merchant_name, transaction_date,
                          purchased_items, total_amount, currency,
                          expense_category):
    """Update an existing expense without changing its receipt ID or owner."""
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(expenses)")
        table_info = cursor.fetchall()
        column_names = {row[1] for row in table_info}

        # Support the existing database table name/schema used by the app.
        if not column_names:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            candidate = next(
                (name for name in table_names if name.lower() in
                 ("expenses", "expense", "user_expenses")), None
            )
            if candidate:
                safe_table = '"' + candidate.replace('"', '""') + '"'
                cursor.execute(f"PRAGMA table_info({safe_table})")
                table_info = cursor.fetchall()
                column_names = {row[1] for row in table_info}
                table_name = candidate
            else:
                raise RuntimeError("Could not find the expenses table.")
        else:
            table_name = "expenses"

        required = {
            "receipt_id", "user_id", "merchant_name", "transaction_date",
            "purchased_items", "total_amount", "expense_category"
        }
        if not required.issubset(column_names):
            raise RuntimeError(
                "The expenses table columns do not match the expected schema."
            )

        assignments = [
            "merchant_name = ?",
            "transaction_date = ?",
            "purchased_items = ?",
            "total_amount = ?",
            "expense_category = ?",
        ]
        values = [
            merchant_name.strip(),
            transaction_date.isoformat(),
            purchased_items.strip(),
            round(float(total_amount), 2),
            expense_category,
        ]
        if "currency" in column_names:
            assignments.append("currency = ?")
            values.append(currency)
        values.extend([receipt_id, user_id])

        safe_table = '"' + table_name.replace('"', '""') + '"'
        cursor.execute(
            f"UPDATE {safe_table} SET {', '.join(assignments)} "
            "WHERE receipt_id = ? AND user_id = ?",
            values,
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Expense Classification System",
    page_icon="💰",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_expense_classifier.joblib"

# --------------------------------------------------
# 2. Initialise database and load model
# --------------------------------------------------

initialize_database()

# Initialise login session state
if "user" not in st.session_state:
    st.session_state.user = None

if "auth_page" not in st.session_state:
    st.session_state.auth_page = "Login"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# Authentication interface
# --------------------------------------------------

st.title("💰 Expense Classification System")

if st.session_state.user is None:

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader("Welcome Back")

        login_username = st.text_input(
            "Username",
            key="login_username",
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button("Login", use_container_width=True):
            user = authenticate_user(
                login_username,
                login_password,
            )

            if user is not None:
                st.session_state.user = user

                # Reset the budget popup for the new login session
                st.session_state["budget_alert_shown_this_login"] = False

                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        st.subheader("Create an Account")

        register_username = st.text_input(
            "Choose a username",
            key="register_username",
        )

        register_password = st.text_input(
            "Choose a password",
            type="password",
            key="register_password",
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="confirm_password",
        )

        if st.button("Create Account", use_container_width=True):
            if register_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(
                    register_username,
                    register_password,
                )

                if success:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)

    st.stop()

# --------------------------------------------------
# Main application navigation
# --------------------------------------------------

current_user = st.session_state.user
current_user_id = current_user["user_id"]
username = current_user.get("username", "User")

st.sidebar.markdown("# 💸 Expense Tracker")
st.sidebar.caption(f"Welcome, {username}")
st.sidebar.divider()


# --------------------------------------------------
# UI Theme
# --------------------------------------------------
st.sidebar.subheader("Appearance")

if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Dark"

app_theme = st.sidebar.selectbox(
    "Choose theme",
    options=["Dark", "Light"],
    key="app_theme",
    label_visibility="collapsed",
)

if app_theme == "Dark":
    theme_colors = {
        "background": "#0E1117",
        "paper": "#0E1117",
        "text": "#F1F5F9",
        "muted_text": "#CBD5E1",
        "grid": "rgba(148, 163, 184, 0.20)",
        "tooltip_bg": "#1E293B",
        "tooltip_text": "#F8FAFC",
        "overall": "#F8FAFC",
    }
else:
    theme_colors = {
        "background": "#FFFFFF",
        "paper": "#FFFFFF",
        "text": "#1E293B",
        "muted_text": "#475569",
        "grid": "rgba(100, 116, 139, 0.20)",
        "tooltip_bg": "#FFFFFF",
        "tooltip_text": "#0F172A",
        "overall": "#172554",
    }


# --------------------------------------------------
# Apply the selected theme to the Streamlit interface
# --------------------------------------------------

if app_theme == "Dark":
    ui_bg = "#0E1117"
    sidebar_bg = "#262833"
    input_bg = "#262833"
    header_bg = "#262833"
    text_color = "#F1F5F9"
    muted_color = "#CBD5E1"
    border_color = "#475569"
    hover_bg = "#374151"
    placeholder_color = "#94A3B8"
    alert_bg = "#422B2B"
    alert_text = "#FECACA"
    popup_shadow = "0 4px 12px rgba(0, 0, 0, 0.35)"
else:
    ui_bg = "#FFFFFF"
    sidebar_bg = "#F1F5F9"
    input_bg = "#FFFFFF"
    header_bg = "#F1F5F9"
    text_color = "#1E293B"
    muted_color = "#475569"
    border_color = "#CBD5E1"
    hover_bg = "#F1F5F9"
    placeholder_color = "#64748B"
    alert_bg = "#FEF3C7"
    alert_text = "#92400E"
    popup_shadow = "0 4px 12px rgba(15, 23, 42, 0.12)"

# Load the external CSS file
CSS_PATH = BASE_DIR / "assets" / "style.css"

with open(CSS_PATH, "r", encoding="utf-8") as css_file:
    theme_css = f"<style>\n{css_file.read()}\n</style>"

theme_replacements = {
    "__UI_BG__": ui_bg,
    "__SIDEBAR_BG__": sidebar_bg,
    "__INPUT_BG__": input_bg,
    "__HEADER_BG__": header_bg,
    "__TEXT_COLOR__": text_color,
    "__MUTED_COLOR__": muted_color,
    "__BORDER_COLOR__": border_color,
    "__HOVER_BG__": hover_bg,
    "__PLACEHOLDER_COLOR__": placeholder_color,
    "__ALERT_BG__": alert_bg,
    "__ALERT_TEXT__": alert_text,
    "__POPUP_SHADOW__": popup_shadow,
}

for placeholder, value in theme_replacements.items():
    theme_css = theme_css.replace(placeholder, value)

st.markdown(theme_css, unsafe_allow_html=True)


page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Receipt", "My Expenses", "Financial Reports", "Budget"],
    index=0,
    label_visibility="collapsed",
    key="main_navigation",
)

if page != "Upload Receipt":
    st.session_state.pop("manual_expense_review", None)
    st.session_state.pop("manual_confirmed_category", None)
    st.session_state.pop("share_manual_feedback", None)

st.sidebar.divider()
if st.sidebar.button("Log out", use_container_width=True):
    st.session_state.user = None
    st.rerun()

# Load the signed-in user's real saved transactions.
user_expenses = get_user_expenses(current_user_id)
expenses_df = pd.DataFrame(user_expenses) if user_expenses else pd.DataFrame()

if not expenses_df.empty:
    if "transaction_date" in expenses_df.columns:
        expenses_df["transaction_date"] = pd.to_datetime(
            expenses_df["transaction_date"], errors="coerce"
        )
    if "total_amount" in expenses_df.columns:
        expenses_df["total_amount"] = pd.to_numeric(
            expenses_df["total_amount"], errors="coerce"
        ).fillna(0.0)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
if page == "Dashboard":
    st.title("💰 Expense Dashboard")
    st.write(f"Welcome back, {username}! 👋")
    st.caption("Your personal overview of spending, transactions, and monthly budget.")

    if expenses_df.empty:
        st.info("You haven't saved any expenses yet. Start by uploading a receipt.")
        def go_to_upload_receipt():
            st.session_state["main_navigation"] = "Upload Receipt"


        if st.button(
            "Upload your first receipt",
            type="primary",
            on_click=go_to_upload_receipt,
        ):
            pass
    else:
        today = date.today()
        
        period = st.selectbox(
            "View spending period",
            ["This Month", "Last 3 Months", "This Year", "All Time"],
            key="dashboard_period",
        )

        valid_dates = expenses_df["transaction_date"].notna()
        dated_df = expenses_df[valid_dates].copy()

        if period == "This Month":
            period_start = pd.Timestamp(today.year, today.month, 1)
            period_df = dated_df[
                dated_df["transaction_date"] >= period_start
            ].copy()

        elif period == "Last 3 Months":
            period_start = pd.Timestamp(today.year, today.month, 1) - pd.DateOffset(months=2)
            period_df = dated_df[
                dated_df["transaction_date"] >= period_start
            ].copy()

        elif period == "This Year":
            period_start = pd.Timestamp(today.year, 1, 1)
            period_df = dated_df[
                dated_df["transaction_date"] >= period_start
            ].copy()

        else:
            period_df = expenses_df.copy()

        total_spending = float(period_df["total_amount"].sum())
        transaction_count = len(period_df)
        average_expense = (
            total_spending / transaction_count
            if transaction_count else 0.0
        )

        month_start = pd.Timestamp(today.year, today.month, 1)
        month_df = dated_df[
            dated_df["transaction_date"] >= month_start
        ].copy()
        month_spending = float(month_df["total_amount"].sum())

        budget = get_monthly_budget(
            current_user_id, today.strftime("%Y-%m")
        )
        budget = float(budget) if budget is not None else 0.0
        remaining = budget - month_spending
        budget_used = month_spending / budget if budget > 0 else 0.0


        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Spending", f"RM{total_spending:,.2f}")
        k2.metric("Transactions", f"{transaction_count}")
        k3.metric("Average Expense", f"RM{average_expense:,.2f}")
        k4.metric("This Month", f"RM{month_spending:,.2f}")

        # --------------------------------------------------
        # Dashboard charts and category ranking
        # --------------------------------------------------
        import plotly.express as px

        category_colors = {
            "Food and Beverage": "#F97316",
            "Groceries": "#16A34A",
            "Transportation": "#0EA5E9",
            "Shopping": "#8B5CF6",
            "Health & Personal Care": "#EC4899",
        }

        # Prepare the selected-period data.
        chart_df = period_df.copy()
        chart_df = chart_df.dropna(
            subset=["transaction_date"]
        )

        if not chart_df.empty:
            chart_df["Day"] = (
                chart_df["transaction_date"].dt.strftime("%Y-%m-%d")
            )
            chart_df["Month"] = (
                chart_df["transaction_date"]
                .dt.to_period("M")
                .astype(str)
            )

        # --------------------------------------------------
        # 1. Donut chart and ranking side by side
        # --------------------------------------------------
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Spending by Category")

            if (
                not period_df.empty
                and "expense_category" in period_df.columns
            ):
                category_totals = (
                    period_df.groupby("expense_category")["total_amount"]
                    .sum()
                    .reset_index()
                )

                category_totals = category_totals[
                    category_totals["total_amount"] > 0
                ]

                if not category_totals.empty:
                    fig_donut = px.pie(
                        category_totals,
                        names="expense_category",
                        values="total_amount",
                        hole=0.55,
                        color="expense_category",
                        color_discrete_map=category_colors,
                    )

                    fig_donut.update_traces(
                        textposition="inside",
                        textinfo="percent",
                        textfont=dict(
                            color="#FFFFFF",
                            size=12,
                        ),
                        insidetextfont=dict(
                            color="#FFFFFF",
                        ),
                        hovertemplate=(
                            "<b>%{label}</b><br>"
                            "Spending: RM%{value:,.2f}<br>"
                            "Share: %{percent}"
                            "<extra></extra>"
                        ),
                    )


                    
                    fig_donut.update_layout(
                        showlegend=True,
                        legend_title_text="Category",
                        height=370,
                        margin=dict(t=20, b=20, l=10, r=10),
                        template="plotly_dark" if app_theme == "Dark" else "plotly_white",
                        paper_bgcolor=theme_colors["paper"],
                        plot_bgcolor=theme_colors["background"],
                        font=dict(
                            color="#F1F5F9" if app_theme == "Dark" else "#1E293B"
                        ),
                        legend=dict(
                            font=dict(
                                color="#F1F5F9" if app_theme == "Dark" else "#1E293B",
                                size=12,
                            ),
                            title=dict(
                                text="Category",
                                font=dict(
                                    color="#F1F5F9" if app_theme == "Dark" else "#1E293B",
                                    size=12,
                                ),
                            ),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                        ),
                    )


                    st.plotly_chart(
                        fig_donut,
                        use_container_width=True,
                    )
                else:
                    st.info("No positive spending for this period.")
            else:
                st.info("No category data is available.")

        with right:
            st.subheader("🏆 Top Spending Categories")

            if (
                not period_df.empty
                and "expense_category" in period_df.columns
            ):
                ranking = (
                    period_df.groupby("expense_category")["total_amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .reset_index()
                )

                ranking.columns = [
                    "Category",
                    "Total Spending (RM)",
                ]

                if not ranking.empty:
                    ranking.insert(
                        0,
                        "Rank",
                        range(1, len(ranking) + 1),
                    )

                    ranking_display = ranking.copy()
                    ranking_display["Total Spending (RM)"] = ranking_display[
                        "Total Spending (RM)"
                    ].map(lambda value: f"RM{float(value):,.2f}")

                    # Display a theme-aware ranking table

                    table_bg = "#FFFFFF" if app_theme == "Light" else "#0E1117"
                    header_bg = "#F1F5F9" if app_theme == "Light" else "#262833"
                    table_text = "#1E293B" if app_theme == "Light" else "#F1F5F9"
                    border_color = "#CBD5E1" if app_theme == "Light" else "#374151"

                    rows_html = ""

                    for _, row in ranking_display.iterrows():
                        category = html.escape(str(row["Category"]))
                        amount = html.escape(str(row["Total Spending (RM)"]))
                        rank = int(row["Rank"])

                        rows_html += f"""
                        <tr>
                            <td style="padding: 9px 10px; text-align: right; border-bottom: 1px solid {border_color};">{rank}</td>
                            <td style="padding: 9px 10px; border-bottom: 1px solid {border_color};">{category}</td>
                            <td style="padding: 9px 10px; border-bottom: 1px solid {border_color};">{amount}</td>
                        </tr>
                        """

                    table_html = f"""
                    <table style="
                        width: 100%;
                        border-collapse: collapse;
                        background-color: {table_bg};
                        color: {table_text};
                        font-size: 14px;
                    ">
                        <thead>
                            <tr style="background-color: {header_bg};">
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid {border_color};">Rank</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid {border_color};">Category</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 1px solid {border_color};">Total Spending (RM)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                    """

                    st.html(table_html)                  

                    highest_category = ranking.iloc[0]

                    st.metric(
                        "Highest-Spending Category",
                        highest_category["Category"],
                        f"RM{highest_category['Total Spending (RM)']:,.2f}",
                    )
                else:
                    st.info("No category spending is available.")
            else:
                st.info("Save expenses to view category rankings.")

        
        # --------------------------------------------------
        # 2. Interactive Spending Trend Comparison
        # --------------------------------------------------
    
        st.divider()
        st.subheader("📈 Spending Trend Comparison")
        st.caption(
            "Click a category in the legend to highlight it. "
            "Other categories turn grey, while Overall Spending "
            "remains visible."
        )

        if (
            not chart_df.empty
            and "expense_category" in chart_df.columns
        ):
            # Trends follow the dashboard's existing spending-period filter.
            # Use short labels without repeating the year on the x-axis.
            chart_df["Month"] = chart_df["transaction_date"].dt.strftime("%b")
            chart_df["Day"] = chart_df["transaction_date"].dt.strftime("%b %d")

            # Calculate spending by month and day.
            monthly_totals = (
                chart_df.groupby(
                    ["Month", "expense_category"]
                )["total_amount"]
                .sum()
                .reset_index()
            )

            daily_totals = (
                chart_df.groupby(
                    ["Day", "expense_category"]
                )["total_amount"]
                .sum()
                .reset_index()
            )

            monthly_overall = (
                chart_df.groupby("Month")["total_amount"]
                .sum()
                .sort_index()
            )

            daily_overall = (
                chart_df.groupby("Day")["total_amount"]
                .sum()
                .sort_index()
            )

            # Keep all known categories in their original colours.
            categories = [
                "Food and Beverage",
                "Groceries",
                "Transportation",
                "Shopping",
                "Health & Personal Care",
            ]

            # Create the chart data for the browser.
            def build_trend_data(category_totals, overall, x_values):
                traces = []

                # Overall Spending is always visible.
                traces.append({
                    "x": list(overall.index),
                    "y": [float(v) for v in overall.values],
                    "name": "Overall Spending",
                    "color": theme_colors["overall"],
                    "overall": True,
                })

                for category in categories:
                    category_data = (
                        category_totals[
                            category_totals["expense_category"] == category
                        ]
                        .set_index(
                            "Month" if x_values == "Month" else "Day"
                        )["total_amount"]
                        .reindex(overall.index, fill_value=0)
                    )

                    traces.append({
                        "x": list(overall.index),
                        "y": [float(v) for v in category_data.values],
                        "name": category,
                        "color": category_colors.get(
                            category, "#64748B"
                        ),
                        "overall": False,
                    })

                return traces

            # Reusable interactive chart.
            def render_highlight_chart(
                chart_id,
                title,
                traces,
                x_title,
            ):
                chart_json = json.dumps(traces)

                html = f"""
                <div style="width:100%;font-family:Arial,sans-serif;">
                    <div id="{chart_id}"
                         style="width:100%;height:450px;"></div>
                </div>

                <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
                <script>
                const chartData = {chart_json};
                const chartElement = document.getElementById("{chart_id}");

                const originalColors = chartData.map(
                    trace => trace.color
                );

                const plotTraces = chartData.map(trace => ({{
                    x: trace.x,
                    y: trace.y,
                    type: "scatter",
                    mode: "lines+markers",
                    name: trace.name,
                    line: {{
                        color: trace.color,
                        width: trace.overall ? 4 : 2.5
                    }},
                    marker: {{
                        color: trace.color,
                        size: trace.overall ? 7 : 5
                    }},
                    connectgaps: false,
                    hovertemplate:
                        "<b>" + trace.name + "</b><br>" +
                        "{x_title}: %{{x}}<br>" +
                        "Spending: RM%{{y:,.2f}}" +
                        "<extra></extra>"
                }}));

                const layout = {{
                    title: {{
                        text: "{title}",
                        x: 0.02,
                        font: {{color: "{theme_colors['text']}"}}
                    }},
                    autosize: true,
                    height: 450,
                    margin: {{t: 55, b: 65, l: 65, r: 190}},
                    paper_bgcolor: "{theme_colors['paper']}",
                    plot_bgcolor: "{theme_colors['background']}",
                    font: {{color: "{theme_colors['text']}"}},
                    xaxis: {{
                        title: "",
                        type: "category",
                        tickangle: 0,
                        automargin: true,
                        categoryorder: "array",
                        categoryarray: chartData[0].x,
                        color: "{theme_colors['muted_text']}",
                        tickfont: {{
                            color: "{theme_colors['muted_text']}",
                            size: 12
                        }},
                        gridcolor: "{theme_colors['grid']}",
                        zerolinecolor: "{theme_colors['grid']}"
                    }},
                    yaxis: {{
                        title: {{
                            text: "Spending (RM)",
                            font: {{
                                color: "{theme_colors['text']}"
                            }}
                        }},
                        rangemode: "tozero",
                        color: "{theme_colors['muted_text']}",
                        tickfont: {{
                            color: "{theme_colors['muted_text']}",
                            size: 12
                        }},
                        gridcolor: "{theme_colors['grid']}",
                        zerolinecolor: "{theme_colors['grid']}"
                    }},

                    hovermode: "x unified",
                    hoverlabel: {{
                        bgcolor: "{theme_colors['tooltip_bg']}",
                        font: {{
                            color: "{theme_colors['tooltip_text']}",
                            size: 13,
                            family: "Arial, sans-serif"
                        }},
                        bordercolor: "{theme_colors['grid']}",
                        align: "left",
                        namelength: -1
                    }},

                    legend: {{
                        title: {{
                            text: "Click a category",
                            font: {{color: "{theme_colors['text']}"}}
                        }},
                        font: {{
                            color: "{theme_colors['text']}",
                            size: 12
                        }},
                        orientation: "v",
                        x: 1.02,
                        xanchor: "left",
                        y: 1,
                        yanchor: "top"
                    }}
                }};


                Plotly.newPlot(
                    chartElement,
                    plotTraces,
                    layout,
                    {{responsive: true, displaylogo: false}}
                ).then(function() {{
                    let selectedCategory = null;

                    chartElement.on(
                        "plotly_legendclick",
                        function(eventData) {{
                            const clickedTrace =
                                chartData[eventData.curveNumber];

                            // Never hide or grey out Overall Spending.
                            if (clickedTrace.overall) {{
                                return false;
                            }}

                            // Clicking the selected category again
                            // restores all original colours.
                            selectedCategory =
                                selectedCategory === clickedTrace.name
                                ? null
                                : clickedTrace.name;

                            const updatedTraces = chartData.map(
                                function(trace) {{
                                    const isOverall = trace.overall;
                                    const isSelected =
                                        trace.name === selectedCategory;

                                    const color =
                                        isOverall || !selectedCategory ||
                                        isSelected
                                        ? trace.color
                                        : "#9CA3AF";

                                    return {{
                                        line: {{color: color}},
                                        marker: {{color: color}},
                                        opacity: 1
                                    }};
                                }}
                            );

                            Plotly.restyle(
                                chartElement,
                                {{
                                    "line.color": updatedTraces.map(
                                        item => item.line.color
                                    ),
                                    "marker.color": updatedTraces.map(
                                        item => item.marker.color
                                    ),
                                    "opacity": updatedTraces.map(
                                        item => item.opacity
                                    )
                                }}
                            );

                            // Prevent Plotly's default hide/show action.
                            return false;
                        }}
                    );
                }});
                </script>
                """

                components.html(html, height=470, scrolling=False)

            # Monthly trend chart.
            st.subheader("Monthly Spending Trend")

            monthly_traces = build_trend_data(
                monthly_totals,
                monthly_overall,
                "Month",
            )

            render_highlight_chart(
                "monthly_spending_highlight",
                "Monthly Spending Trend",
                monthly_traces,
                "Month",
            )

            # Daily trend chart.
            st.subheader("Daily Spending Trend")

            daily_traces = build_trend_data(
                daily_totals,
                daily_overall,
                "Day",
            )

            render_highlight_chart(
                "daily_spending_highlight",
                "Daily Spending Trend",
                daily_traces,
                "Date",
            )

        else:
            st.info(
                "Valid transaction dates and category data are needed "
                "to display the spending trends."
            )


        # --------------------------------------------------
        # 5. Recent transactions and monthly budget
        # --------------------------------------------------
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.subheader("Recent Transactions")

            recent_cols = [
                c for c in [
                    "transaction_date",
                    "merchant_name",
                    "expense_category",
                    "total_amount",
                ]
                if c in expenses_df.columns
            ]

            recent = (
                expenses_df.sort_values(
                    "transaction_date",
                    ascending=False,
                    na_position="last",
                )
                if "transaction_date" in expenses_df.columns
                else expenses_df
            )

            recent_display = recent[recent_cols].head(5).copy()
            if "total_amount" in recent_display.columns:
                recent_display["total_amount"] = recent_display[
                    "total_amount"
                ].map(lambda value: f"RM{float(value):,.2f}")
            render_styled_table(
                readable_column_names(recent_display),
                app_theme,
            )

        with c2:
            st.subheader("Monthly Budget")

            if budget > 0:
                st.metric("Budget", f"RM{budget:,.2f}")
                st.metric("Remaining", f"RM{remaining:,.2f}")
                
                if budget_used >= 1.0:
                    st.error(
                        f"🚨 Budget exceeded by RM{abs(remaining):,.2f} "
                        f"({budget_used:.1%} of budget used)."
                    )
                    st.progress(min(budget_used, 1.0))

                elif budget_used >= 0.7:
                    st.error(
                        f"⚠️ Budget warning: {budget_used:.1%} used. "
                        f"RM{remaining:,.2f} remaining."
                    )
                    st.progress(budget_used)

                else:
                    st.progress(budget_used)
                    st.caption(
                        f"{budget_used:.1%} of this month's budget used"
                    )

            else:
                st.info(
                    "Set a monthly budget in the Budget page to track progress."
                )


# --------------------------------------------------
# Upload receipt (existing workflow retained)
# --------------------------------------------------
elif page == "Upload Receipt":
    # 5. Receipt Upload, Classification and Saving
    # --------------------------------------------------

    st.title("🧾 Upload Receipt")
    st.caption(
        "Upload your receipt, review the extracted information, "
        "predict the expense category, and save your transaction."
    )

    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="receipt_upload",
    )

    if uploaded_file is not None:

        # Identify the uploaded receipt so old details are not reused.
        current_signature = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        previous_signature = st.session_state.get(
            "previous_uploaded_file"
        )

        if previous_signature != current_signature:

            # Remove old receipt data and widget values.
            keys_to_remove = [
                "receipt_gemini_text",
                "receipt_details",
                "receipt_prediction",
                "reviewed_expense",
                "confirmed_expense_category",
                "gemini_used_model",
                "gemini_extraction_seconds",
                "gemini_attempt_timings",
                "receipt_prediction_failed",
                "share_receipt_feedback",
            ]

            for key in list(st.session_state.keys()):
                if (
                    key in keys_to_remove
                    or key.startswith("merchant_candidate_selector_")
                    or key.startswith("merchant_name_review_")
                    or key.startswith("transaction_date_review_")
                    or key.startswith("purchased_items_review_")
                    or key.startswith("total_amount_review_")
                    or key.startswith("expense_review_form_")
                ):
                    st.session_state.pop(key, None)

            st.session_state["previous_uploaded_file"] = (
                current_signature
            )

        # Unique widget keys for the currently uploaded receipt.
        dropdown_key = (
            f"merchant_candidate_selector_{current_signature}"
        )
        merchant_key = f"merchant_name_review_{current_signature}"
        date_key = f"transaction_date_review_{current_signature}"
        items_key = f"purchased_items_review_{current_signature}"
        amount_key = f"total_amount_review_{current_signature}"
        form_key = f"expense_review_form_{current_signature}"

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded receipt",
                use_container_width=True,
            )


        
            # --------------------------------------------------
            # Extract receipt text
            # --------------------------------------------------

            
            if st.button(
                "Extract Receipt Text",
                key=f"extract_receipt_{current_signature}",
            ):
                try:
                    with st.spinner("Gemini is analysing your receipt..."):
                        prompt = """
You are extracting information from a receipt image for an
academic expense classification system.

Read the receipt image directly. Do not guess or invent missing
information. Return valid JSON only, without Markdown or code fences.

Use exactly this JSON structure:
{
  "merchant_name": "",
  "transaction_date": "",
  "purchased_items": [
    {
      "item_code": "",
      "description": "",
      "quantity": "",
      "unit_price": "",
      "discount": "",
      "total_price": ""
    }
  ],
  "total_amount": "",
  "currency": ""
}

Rules:
1. merchant_name: the store or business printed on the receipt,
   not the customer's name.
2. transaction_date: use YYYY-MM-DD only when the date is clear.
   Otherwise return an empty string.
3. purchased_items: extract each visible purchased item separately.
   Preserve item codes separately from descriptions.
4. quantity, unit_price, discount and total_price must be strings.
   Do not calculate or invent missing values.
5. total_amount: extract the final transaction total, not the
   subtotal, amount paid, or change, when these are separately shown.
6. currency: use a currency code such as MYR, SGD or USD only when
   the receipt provides enough evidence. Otherwise return an
   empty string.
7. Use empty strings for missing text values and [] when no
   purchased items can be identified.
8. Return valid JSON only.
"""
                        models_to_try = [
                            ("gemini-flash-lite-latest", 15000),
                            ("gemini-3.1-flash-lite", 10000),
                        ]

                        response = None
                        used_model = None
                        attempt_timings = []

                        for model_name, timeout_ms in models_to_try:
                            start_time = time.perf_counter()

                            try:
                                # Give each model its own request timeout.
                                request_client = genai.Client(
                                    api_key=GEMINI_API_KEY,
                                    http_options=types.HttpOptions(
                                        timeout=timeout_ms,
                                        retry_options=types.HttpRetryOptions(
                                            attempts=1
                                        ),
                                    ),
                                )

                                response = request_client.models.generate_content(
                                    model=model_name,
                                    contents=[prompt, image],
                                )

                                elapsed = time.perf_counter() - start_time

                                attempt_timings.append({
                                    "model": model_name,
                                    "seconds": elapsed,
                                    "status": "Response received",
                                })

                                used_model = model_name
                                break

                            except Exception:
                                elapsed = time.perf_counter() - start_time

                                attempt_timings.append({
                                    "model": model_name,
                                    "seconds": elapsed,
                                    "status": "Failed",
                                })
                        
                        
                        if response is None:
                            st.error(
                                "Receipt extraction could not be completed."
                            )
                            st.info(
                                "Please try again. If the problem continues, "
                                "check your internet connection and try again later."
                            )
                            st.stop()

                        extraction_seconds = sum(
                            attempt["seconds"]
                            for attempt in attempt_timings
                        )

                        raw_response = (response.text or "").strip()

                        # Remove code fences if Gemini includes them.
                        raw_response = re.sub(
                            r"^```(?:json)?\s*|\s*```$",
                            "",
                            raw_response,
                            flags=re.IGNORECASE,
                        ).strip()

                        details_from_gemini = json.loads(raw_response)

                        merchant = str(
                            details_from_gemini.get("merchant_name") or ""
                        ).strip()

                        date_text = str(
                            details_from_gemini.get("transaction_date") or ""
                        ).strip()

                        items_data = (
                            details_from_gemini.get("purchased_items") or []
                        )

                        # Convert the item-by-item JSON into readable text
                        # for the existing review form and classifier.
                        item_lines = []

                        for item in items_data:
                            if not isinstance(item, dict):
                                continue

                            parts = []

                            item_code = str(item.get("item_code") or "").strip()
                            description = str(
                                item.get("description") or ""
                            ).strip()
                            quantity = str(item.get("quantity") or "").strip()
                            unit_price = str(
                                item.get("unit_price") or ""
                            ).strip()
                            discount = str(item.get("discount") or "").strip()
                            item_total = str(
                                item.get("total_price") or ""
                            ).strip()

                            if item_code:
                                parts.append(f"Code: {item_code}")
                            if description:
                                parts.append(description)
                            if quantity:
                                parts.append(f"Qty: {quantity}")
                            if unit_price:
                                parts.append(f"Unit price: {unit_price}")
                            if discount:
                                parts.append(f"Discount: {discount}")
                            if item_total:
                                parts.append(f"Total: {item_total}")

                            if parts:
                                item_lines.append(" | ".join(parts))

                        items_text = "\n".join(item_lines)

                        # Parse the final total into a number for the
                        # existing amount input.
                        amount_text = str(
                            details_from_gemini.get("total_amount") or ""
                        ).strip()

                        amount_match = re.search(
                            r"-?\d[\d,]*(?:\.\d+)?",
                            amount_text,
                        )

                        total_amount = (
                            float(amount_match.group(0).replace(",", ""))
                            if amount_match
                            else None
                        )

                        # Use the extracted date only if it is valid ISO format.
                        # The form defaults to today's date when no valid date
                        # can be extracted; the stored extraction remains blank.
                        if date_text:
                            try:
                                parsed_date = pd.to_datetime(
                                    date_text,
                                    format="%Y-%m-%d",
                                    errors="raise",
                                ).date()
                            except (ValueError, TypeError):
                                date_text = ""
                                parsed_date = date.today()
                        else:
                            parsed_date = date.today()

                        receipt_details = {
                            "merchant_name": merchant,
                            "merchant_candidates": [merchant] if merchant else [""],
                            "transaction_date": date_text,
                            "purchased_items": items_text,
                            "total_amount": total_amount,
                            "currency": str(
                                details_from_gemini.get("currency") or ""
                            ).strip(),
                        }

                        # Keep the session-state names expected by the
                        # existing receipt review form.
                        st.session_state["receipt_gemini_text"] = json.dumps(
                            details_from_gemini,
                            ensure_ascii=False,
                            indent=2,
                        )
                        st.session_state["receipt_details"] = receipt_details

                        st.session_state[dropdown_key] = merchant
                        st.session_state[merchant_key] = merchant
                        st.session_state[date_key] = parsed_date
                        st.session_state[items_key] = items_text
                        st.session_state[amount_key] = (
                            round(total_amount, 2)
                            if total_amount is not None
                            else 0.0
                        )

                        # Clear previous predictions and review results.
                        for key in [
                            "predicted_category",
                            "confirmed_category",
                            "prediction_result",
                            "review_complete",
                            "receipt_prediction",
                            "receipt_prediction_failed",
                            "reviewed_expense",
                            "confirmed_expense_category",
                            "share_receipt_feedback",
                        ]:
                            st.session_state.pop(key, None)
                                          
                        # Save extraction information before rerunning the page.
                        st.session_state["gemini_used_model"] = used_model
                        st.session_state["gemini_extraction_seconds"] = extraction_seconds
                        st.session_state["gemini_attempt_timings"] = attempt_timings

                        st.success("Receipt details extracted successfully!")
                        st.rerun()


                except json.JSONDecodeError:
                    st.error(
                        "The receipt details could not be read correctly. "
                        "Please try extracting the receipt again."
                    )

                except Exception:
                    st.error("Receipt extraction encountered a problem. Please try again.")


        
            # --------------------------------------------------
            # Display Gemini extraction results
            # --------------------------------------------------

            extracted_text = st.session_state.get(
                "receipt_gemini_text", ""
            )

            receipt_details = st.session_state.get(
                "receipt_details", {}
            )

            if extracted_text:

                st.subheader("Gemini Extraction Result")
                
                used_model = st.session_state.get("gemini_used_model")
                extraction_seconds = st.session_state.get(
                    "gemini_extraction_seconds"
                )
                attempt_timings = st.session_state.get(
                    "gemini_attempt_timings", []
                )

                if used_model and extraction_seconds is not None:
                    col1, col2 = st.columns(2)

                    col1.metric("Gemini Model Used", used_model)
                    col2.metric(
                        "Extraction Time",
                        f"{extraction_seconds:.2f}s",
                    )

                    if len(attempt_timings) > 1:
                        st.info(
                            "The primary model failed, so the fallback model "
                            "was used successfully."
                        )

                st.text_area(
                    "Extracted receipt details (for reference)",
                    value=extracted_text,
                    height=180,
                    disabled=True,
                    key=f"gemini_result_{current_signature}",
                )

                st.subheader("Review Expense Details")

                merchant_candidates = receipt_details.get(
                    "merchant_candidates", []
                )

                suggested_merchant = (
                    merchant_candidates[0]
                    if merchant_candidates
                    else receipt_details.get("merchant_name", "")
                )

                if not merchant_candidates:
                    merchant_candidates = [suggested_merchant]

                # Ensure the selected value is one of the options.
                if st.session_state.get(dropdown_key) not in (
                    merchant_candidates
                ):
                    st.session_state[dropdown_key] = (
                        suggested_merchant
                    )

                # Keep the dropdown and editable merchant field linked.
            
                def update_merchant_name():
                    selected = st.session_state.get(
                        dropdown_key,
                        suggested_merchant,
                    )
                    st.session_state[merchant_key] = selected

            
                if dropdown_key not in st.session_state:
                    st.session_state[dropdown_key] = suggested_merchant

                if merchant_key not in st.session_state:
                    st.session_state[merchant_key] = suggested_merchant 
            
                st.selectbox(
                    "Merchant Name Suggestions",
                    options=merchant_candidates,
                    key=dropdown_key,
                    on_change=update_merchant_name,
                    help=(
                        "Choose a suggestion. The editable merchant "
                        "field below will update automatically."
                    ),
                )


                # This input is OUTSIDE the form so the dropdown can
                # update it immediately, and you can still edit it.
                merchant_name = st.text_input(
                    "Edit Merchant Name (if needed)",
                    key=merchant_key,
                    help=(
                        "You can correct the selected merchant name here."
                    ),
                )

                detected_date = receipt_details.get(
                    "transaction_date", ""
                )

                try:
                    suggested_date = (
                        pd.to_datetime(
                            detected_date,
                            dayfirst=True,
                        ).date()
                        if detected_date
                        else date.today()
                    )
                except (ValueError, TypeError):
                    suggested_date = date.today()

                detected_amount = receipt_details.get("total_amount")

                # --------------------------------------------------
                # Review the other fields and predict the category
                # --------------------------------------------------

                with st.form(form_key):

                    transaction_date = st.date_input(
                        "Transaction Date",
                        value=st.session_state.get(
                            date_key, suggested_date
                        ),
                        key=date_key,
                    )

                    purchased_items = st.text_area(
                        "Purchased Items",
                        value=receipt_details.get(
                            "purchased_items", ""
                        ),
                        key=items_key,
                        height=120,
                    )

                    total_amount = st.number_input(
                        "Total Amount (RM)",
                        min_value=0.0,
                        value=(
                            float(detected_amount)
                            if detected_amount is not None
                            else 0.0
                        ),
                        step=0.10,
                        format="%.2f",
                        key=amount_key,
                    )

                    predict_clicked = st.form_submit_button(
                        "Predict Expense Category"
                    )

                if predict_clicked:

                    if model is None:
                        st.error(
                            "The classification model is unavailable."
                        )

                    elif (
                        not merchant_name.strip()
                        and not purchased_items.strip()
                    ):
                        st.warning(
                            "Enter a merchant name or purchased items first."
                        )

                    elif total_amount <= 0:
                        st.warning(
                            "Please enter a valid total amount."
                        )

                    else:
                        
                        model_text = (
                            merchant_name.strip()
                            + " "
                            + purchased_items.strip()
                        ).strip()
                        
                        feedback_prediction = predict_from_feedback(
                            merchant_name,
                            purchased_items,
                        )

                        if feedback_prediction is not None:
                            prediction = feedback_prediction
                        else:
                            vectorizer = model.named_steps["tfidf"]
                            input_vector = vectorizer.transform([model_text])

                            if input_vector.nnz == 0:
                                prediction = None
                            else:
                                prediction = str(model.predict([model_text])[0])

                        st.session_state["receipt_prediction"] = prediction
                        st.session_state["receipt_prediction_failed"] = (
                            prediction is None
                        )

                        # Save the submitted values for the Save button.
                        st.session_state["reviewed_expense"] = {
                            "merchant_name": merchant_name.strip(),
                            "transaction_date": (
                                transaction_date.isoformat()
                            ),
                            "purchased_items": purchased_items.strip(),
                            "total_amount": round(
                                float(total_amount), 2
                            ),
                        }

                        st.rerun()

                # --------------------------------------------------
                # Confirm the category and save the expense
                # --------------------------------------------------

                prediction = st.session_state.get(
                    "receipt_prediction"
                )

                prediction_failed = st.session_state.get(
                    "receipt_prediction_failed",
                    False,
                )

                if prediction or prediction_failed:
                    if prediction:
                        st.success(f"Predicted category: {prediction}")
                    else:
                        st.warning(
                            "We couldn't automatically classify this expense. "
                            "Please select the correct category manually."
                        )

                    categories = [
                        "Food and Beverage",
                        "Groceries",
                        "Transportation",
                        "Shopping",
                        "Health & Personal Care",
                    ]

                    default_index = (
                        categories.index(prediction)
                        if prediction in categories
                        else 0
                    )

                    confirmed_category = st.selectbox(
                        "Confirm or change the category",
                        options=categories,
                        index=default_index,
                        key="confirmed_expense_category",
                    )
                    
                    share_feedback = st.checkbox(
                        "Allow this category example to improve predictions for other users",
                        value=False,
                        key="share_receipt_feedback",
                    )

                    if st.button(
                        "Save Expense",
                        key=f"save_expense_{current_signature}",
                    ):
                        reviewed = st.session_state.get(
                            "reviewed_expense", {}
                        )

                        if not reviewed.get("merchant_name"):
                            st.error("Please enter the merchant name.")

                        elif not reviewed.get("purchased_items"):
                            st.error("Please enter the purchased items.")

                        elif reviewed.get("total_amount", 0) <= 0:
                            st.error(
                                "Please enter a valid total amount."
                            )

                        else:
                            try:
                                save_expense(
                                    user_id=st.session_state.user["user_id"],
                                    receipt_id=str(uuid.uuid4()),
                                    merchant_name=reviewed["merchant_name"],
                                    transaction_date=reviewed["transaction_date"],
                                    purchased_items=reviewed["purchased_items"],
                                    total_amount=reviewed["total_amount"],
                                    currency="MYR",
                                    expense_category=confirmed_category,
                                    predicted_category=prediction or "Manual selection",
                                )
              
                                save_learning_feedback(
                                    user_id=st.session_state.user["user_id"],
                                    merchant_name=reviewed["merchant_name"],
                                    purchased_items=reviewed["purchased_items"],
                                    predicted_category=prediction or "Manual selection",
                                    confirmed_category=confirmed_category,
                                    share_for_learning=share_feedback,
                                )

                                st.success(
                                    "Expense saved successfully!"
                                )

                                for key in [
                                    "receipt_gemini_text",
                                    "receipt_details",
                                    "receipt_prediction",
                                    "receipt_prediction_failed",
                                    "reviewed_expense",
                                    "confirmed_expense_category",
                                    "share_receipt_feedback",
                                ]:
                                    st.session_state.pop(key, None)

                                st.rerun()

                            except Exception:
                                st.error(
                                    "Could not save the expense. Please try again."
                                )

            elif "receipt_gemini_text" in st.session_state:
                st.warning(
                    "No text was detected. Try a clearer receipt image."
                )

        except Exception:
            st.error(
                "Something went wrong while processing the receipt. "
                "Please try again."
            )
    
    # --------------------------------------------------
    # Manual Expense Entry (for purchases without receipts)
    # --------------------------------------------------
    st.divider()
    st.subheader("✍️ Enter Expense Manually")
    st.caption(
        "No receipt? Enter your purchase details below "
        "to predict its expense category."
    )

    with st.form("manual_expense_form"):
        manual_merchant = st.text_input(
            "Merchant / Store Name",
            key="manual_merchant",
            placeholder="e.g. McDonald's"
        )

        manual_date = st.date_input(
            "Transaction Date",
            value=date.today(),
            key="manual_date"
        )

        manual_items = st.text_area(
            "Purchased Items",
            key="manual_items",
            placeholder="e.g. Burger, fries and iced tea"
        )

        manual_amount = st.number_input(
            "Total Amount (RM)",
            min_value=0.01,
            value=1.00,
            step=0.10,
            format="%.2f",
            key="manual_amount"
        )

        manual_predict = st.form_submit_button(
            "Predict Expense Category",
            type="primary"
        )

    if manual_predict:
        if model is None:
            st.error("The classification model is unavailable.")

        elif not manual_merchant.strip():
            st.warning("Please enter the merchant or store name.")

        elif not manual_items.strip():
            st.warning("Please enter the purchased items.")

        else:
            manual_model_text = (
                manual_merchant.strip()
                + " "
                + manual_items.strip()
            ).strip()
            
            # Check shared feedback before using the ML model.
            feedback_prediction = predict_from_feedback(
                manual_merchant,
                manual_items,
            )

            if feedback_prediction is not None:
                manual_prediction = feedback_prediction

            else:
                vectorizer = model.named_steps["tfidf"]
                input_vector = vectorizer.transform([manual_model_text])

                if input_vector.nnz == 0:
                    manual_prediction = None
                else:
                    manual_prediction = str(
                        model.predict([manual_model_text])[0]
                    )

            # Always allow the user to review and classify a valid expense.
            st.session_state["manual_expense_review"] = {
                "merchant_name": manual_merchant.strip(),
                "transaction_date": manual_date.isoformat(),
                "purchased_items": manual_items.strip(),
                "total_amount": round(float(manual_amount), 2),
                "predicted_category": manual_prediction,
            }

            if manual_prediction is not None:
                st.session_state["manual_confirmed_category"] = (
                    manual_prediction
                )
            else:
                st.session_state.pop(
                    "manual_confirmed_category", None
                )


    # Show the prediction after the form has been submitted.
    manual_review = st.session_state.get("manual_expense_review")

    if manual_review:

        if manual_review["predicted_category"]:
            st.success(
                "Predicted category: "
                + manual_review["predicted_category"]
            )
        else:
            st.warning(
                "We couldn't automatically classify this expense. "
                "Please select the correct category manually."
            )

        manual_categories = [
            "Food and Beverage",
            "Groceries",
            "Transportation",
            "Shopping",
            "Health & Personal Care",
        ]

        predicted = manual_review["predicted_category"]

        if st.session_state.get("manual_confirmed_category") not in manual_categories:
            st.session_state["manual_confirmed_category"] = (
                predicted if predicted in manual_categories
                else manual_categories[0]
            )

        manual_confirmed = st.selectbox(
            "Confirm or change the category",
            options=manual_categories,
            key="manual_confirmed_category",
        )

        share_manual_feedback = st.checkbox(
            "Allow this category example to improve predictions for other users",
            value=False,
            key="share_manual_feedback",
        )

        if st.button("Save Manual Expense", type="primary"):
            try:
                save_expense(
                    user_id=current_user_id,
                    receipt_id=str(uuid.uuid4()),
                    merchant_name=manual_review["merchant_name"],
                    transaction_date=manual_review["transaction_date"],
                    purchased_items=manual_review["purchased_items"],
                    total_amount=manual_review["total_amount"],
                    currency="MYR",
                    expense_category=manual_confirmed,
                    predicted_category=(
                        manual_review["predicted_category"]
                        or "Manual selection"
                    ),
                )

                save_learning_feedback(
                    user_id=current_user_id,
                    merchant_name=manual_review["merchant_name"],
                    purchased_items=manual_review["purchased_items"],
                    predicted_category=(
                        manual_review["predicted_category"]
                        or "Manual selection"
                    ),
                    confirmed_category=manual_confirmed,
                    share_for_learning=share_manual_feedback,
                )

                st.success("Manual expense saved successfully!")

                st.session_state.pop("manual_expense_review", None)
                st.session_state.pop("manual_confirmed_category", None)
                st.session_state.pop("share_manual_feedback", None)
                st.rerun()

            except Exception:
                st.error("Could not save the expense. Please try again.")
        


    # --------------------------------------------------

elif page == "My Expenses":
    st.title("📋 My Expenses")
    st.caption("View, filter, edit, download, or delete your saved transactions.")

    if expenses_df.empty:
        st.info("No expenses have been saved yet. Upload a receipt to get started.")
    else:
        category_values = (sorted(expenses_df["expense_category"].dropna().astype(str).unique().tolist())
            if "expense_category" in expenses_df.columns else [])
        chosen_category = st.selectbox(
            "Filter by category",
            ["All Categories"] + category_values,
            key="my_expenses_category_filter",
        )
        filtered = expenses_df.copy()
        if chosen_category != "All Categories" and "expense_category" in filtered.columns:
            filtered = filtered[filtered["expense_category"] == chosen_category]

        date_col1, date_col2 = st.columns(2)
        if "transaction_date" in filtered.columns and filtered["transaction_date"].notna().any():
            min_d = filtered["transaction_date"].dropna().min().date()
            max_d = filtered["transaction_date"].dropna().max().date()
            with date_col1:
                start_date = st.date_input("From", value=min_d, key="my_expenses_start")
            with date_col2:
                end_date = st.date_input("To", value=max_d, key="my_expenses_end")
            if start_date <= end_date:
                filtered = filtered[
                    filtered["transaction_date"].dt.date.between(start_date, end_date)
                ]
            else:
                st.warning("The start date must be on or before the end date.")

        display_cols = [
            c for c in [
                "transaction_date", "merchant_name", "purchased_items",
                "total_amount", "currency", "expense_category", "predicted_category"
            ] if c in filtered.columns
        ]
        display_expenses = filtered[display_cols].copy()
        if "total_amount" in display_expenses.columns:
            display_expenses["total_amount"] = display_expenses[
                "total_amount"
            ].map(lambda value: f"RM{float(value):,.2f}")
        render_styled_table(
            readable_column_names(display_expenses),
            app_theme,
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "Filtered Spending",
            f"RM{filtered['total_amount'].sum():,.2f}"
            if "total_amount" in filtered else "RM0.00",
        )
        m2.metric("Transactions", len(filtered))

        csv_data = filtered[display_cols].copy()
        if "total_amount" in csv_data.columns:
            csv_data["total_amount"] = csv_data["total_amount"].map(
                lambda value: f"{float(value):.2f}"
            )
        csv_data.columns = [
            str(column).replace("_", " ").title()
            for column in csv_data.columns
        ]
        st.download_button(
            "Download filtered expenses (CSV)",
            csv_data.to_csv(index=False).encode("utf-8-sig"),
            "my_expenses.csv",
            "text/csv",
        )

        if "receipt_id" in expenses_df.columns:
            # Use the same category and date filters for the edit/delete choices.
            st.divider()
            st.subheader("Edit an Expense")
            edit_source = filtered
            if edit_source.empty:
                st.info("No transactions match the current filters. Change the filters to edit an expense.")
            else:
                edit_options = {}
                for _, row in edit_source.iterrows():
                    row_date = row.get("transaction_date", "")
                    if pd.notna(row_date):
                        row_date = pd.to_datetime(row_date).strftime("%Y-%m-%d")
                    label = (
                        f"{row_date} | {row.get('merchant_name', '')} | "
                        f"RM{float(row.get('total_amount', 0) or 0):.2f} | "
                        f"{row.get('expense_category', '')}"
                    )
                    edit_options[label] = row["receipt_id"]

                selected_edit_label = st.selectbox(
                    "Choose transaction to edit",
                    options=list(edit_options.keys()),
                    key="edit_expense_choice",
                )
                selected_receipt_id = edit_options[selected_edit_label]
                selected_row = expenses_df[
                    expenses_df["receipt_id"] == selected_receipt_id
                ].iloc[0]

                with st.form("edit_expense_form"):
                    edit_merchant = st.text_input(
                        "Merchant Name",
                        value=str(selected_row.get("merchant_name", "") or ""),
                    )
                    raw_date = selected_row.get("transaction_date")
                    edit_date = st.date_input(
                        "Transaction Date",
                        value=(
                            pd.to_datetime(raw_date).date()
                            if pd.notna(raw_date) else date.today()
                        ),
                    )
                    edit_items = st.text_area(
                        "Purchased Items",
                        value=str(selected_row.get("purchased_items", "") or ""),
                    )
                    edit_amount = st.number_input(
                        "Total Amount (RM)",
                        min_value=0.01,
                        value=max(float(selected_row.get("total_amount", 0) or 0), 0.01),
                        step=0.10,
                        format="%.2f",
                    )
                    edit_categories = [
                        "Food and Beverage", "Groceries", "Transportation",
                        "Shopping", "Health & Personal Care"
                    ]
                    existing_category = str(selected_row.get("expense_category", ""))
                    category_index = (
                        edit_categories.index(existing_category)
                        if existing_category in edit_categories else 0
                    )
                    edit_category = st.selectbox(
                        "Expense Category",
                        edit_categories,
                        index=category_index,
                    )
                    save_edit = st.form_submit_button("Save Changes", type="primary")

                if save_edit:
                    if not edit_merchant.strip():
                        st.error("Please enter a merchant name.")
                    elif not edit_items.strip():
                        st.error("Please enter the purchased items.")
                    elif edit_amount <= 0:
                        st.error("Please enter an amount greater than zero.")
                    else:
                        try:
                            edited = update_expense_record(
                                user_id=current_user_id,
                                receipt_id=selected_receipt_id,
                                merchant_name=edit_merchant,
                                transaction_date=edit_date,
                                purchased_items=edit_items,
                                total_amount=edit_amount,
                                currency=str(selected_row.get("currency", "MYR") or "MYR"),
                                expense_category=edit_category,
                            )
                            if edited:
                                st.success("Expense updated successfully.")
                                st.rerun()
                            else:
                                st.error("The expense could not be found or updated.")
                        except Exception:
                            st.error("Could not update the expense. Please try again.")

            st.divider()
            st.subheader("Delete an Expense")
            if filtered.empty:
                st.info("No transactions match the current filters.")
            else:
                delete_options = {}
                for _, row in filtered.iterrows():
                    row_date = row.get("transaction_date", "")
                    if pd.notna(row_date):
                        row_date = pd.to_datetime(row_date).strftime("%Y-%m-%d")
                    label = (
                        f"{row_date} | {row.get('merchant_name', '')} | "
                        f"RM{float(row.get('total_amount', 0) or 0):.2f} | "
                        f"{row.get('expense_category', '')}"
                    )
                    delete_options[label] = row["receipt_id"]
                selected_delete_label = st.selectbox(
                    "Choose transaction to delete",
                    options=list(delete_options.keys()),
                    key="delete_expense_choice",
                )
                confirm = st.checkbox(
                    "I confirm that I want to delete this expense.",
                    key="confirm_delete_expense",
                )
                if st.button("Delete selected expense", type="secondary"):
                    if not confirm:
                        st.warning("Please confirm before deleting.")
                    elif delete_expense(
                        user_id=current_user_id,
                        receipt_id=delete_options[selected_delete_label],
                    ):
                        st.success("Expense deleted.")
                        st.rerun()
                    else:
                        st.error("The expense could not be deleted.")

# --------------------------------------------------
# Financial reports
# --------------------------------------------------
elif page == "Financial Reports":
    st.title("📊 Financial Reports")
    st.caption(
        "Understand your spending habits through category breakdowns, "
        "monthly trends, and expense summaries."
    )
    if expenses_df.empty:
        st.info("Your reports will appear here after you save expenses.")
    else:
        
        if "expense_category" in expenses_df.columns:
            st.subheader("Spending by Category")
            import plotly.express as px

            by_category = (
                expenses_df.groupby("expense_category")["total_amount"]
                .sum()
                .sort_values(ascending=True)
                .reset_index()
            )
            by_category.columns = ["Expense Category", "Total Spending (RM)"]

            category_colors = {
                "Food and Beverage": "#F97316",
                "Groceries": "#16A34A",
                "Transportation": "#0EA5E9",
                "Shopping": "#8B5CF6",
                "Health & Personal Care": "#EC4899",
            }

            category_fig = px.bar(
                by_category,
                x="Total Spending (RM)",
                y="Expense Category",
                orientation="h",
                text="Total Spending (RM)",
                color="Expense Category",
                color_discrete_map=category_colors,
            )

            category_fig.update_traces(
                texttemplate="RM%{x:,.2f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Spending: RM%{x:,.2f}"
                    "<extra></extra>"
                ),
            )

            # Theme-aware chart text and gridlines
            is_dark = app_theme == "Dark"
            chart_text = "#F3F4F6" if is_dark else "#1F2937"
            chart_muted = "#D1D5DB" if is_dark else "#4B5563"
            chart_grid = "#374151" if is_dark else "#E5E7EB"

            category_fig.update_layout(
                height=max(300, 65 * len(by_category)),
                margin=dict(l=20, r=100, t=20, b=45),
                template="plotly_dark" if is_dark else "plotly_white",
                paper_bgcolor=theme_colors["paper"],
                plot_bgcolor=theme_colors["background"],
                font=dict(
                    family="Arial, sans-serif",
                    size=13,
                    color=chart_text,
                ),
                xaxis=dict(
                    title="Total Spending (RM)",
                    title_font=dict(size=13, color=chart_text),
                    tickfont=dict(size=12, color=chart_muted),
                    tickformat=",.2f",
                    rangemode="tozero",
                    gridcolor=chart_grid,
                    zerolinecolor=chart_grid,
                    showline=False,
                ),
                yaxis=dict(
                    title="",
                    title_font=dict(color=chart_text),
                    tickfont=dict(size=12, color=chart_muted),
                    automargin=True,
                    tickangle=0,
                    showgrid=False,
                ),
                legend=dict(
                    title=dict(text="Expense Category", font=dict(color=chart_text)),
                    font=dict(size=12, color=chart_text),
                ),
            )

            # Ensure the RM labels remain readable in both themes
            category_fig.update_traces(
                selector=dict(type="bar"),
                textfont=dict(color=chart_text, size=12),
            )

            st.plotly_chart(category_fig, use_container_width=True)

            report_table = by_category.copy()
            report_table["Total Spending (RM)"] = report_table[
                "Total Spending (RM)"
            ].map(lambda value: f"RM{float(value):,.2f}")

            render_styled_table(
                readable_column_names(report_table),
                app_theme,
            )
            
        if "transaction_date" in expenses_df.columns:
            trend_df = expenses_df.dropna(subset=["transaction_date"]).copy()

            if not trend_df.empty:
                trend_df["Month Start"] = (
                    trend_df["transaction_date"]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )

                by_month = (
                    trend_df.groupby("Month Start", as_index=False)["total_amount"]
                    .sum()
                    .sort_values("Month Start")
                )

                # Include the year to distinguish months across different years
                by_month["Month"] = by_month["Month Start"].dt.strftime("%b %Y")

                st.subheader("Monthly Spending Trend")

                monthly_report_fig = px.line(
                    by_month,
                    x="Month",
                    y="total_amount",
                    markers=True,
                    text="total_amount",
                )

                is_dark = app_theme == "Dark"
                chart_text = "#F3F4F6" if is_dark else "#1F2937"
                chart_muted = "#D1D5DB" if is_dark else "#4B5563"
                chart_grid = "#374151" if is_dark else "#E5E7EB"

                monthly_report_fig.update_traces(
                    line=dict(
                        color="#38A5F5",
                        width=3,
                    ),
                    marker=dict(
                        color="#38A5F5",
                        size=8,
                        line=dict(
                            color="#38A5F5",
                            width=1,
                        ),
                    ),
                    texttemplate="RM%{y:,.2f}",
                    textposition="top center",
                    textfont=dict(
                        color=chart_text,
                        size=12,
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Spending: RM%{y:,.2f}"
                        "<extra></extra>"
                    ),
                )

                monthly_report_fig.update_layout(
                    height=400,
                    margin=dict(l=65, r=45, t=45, b=65),
                    template="plotly_dark" if is_dark else "plotly_white",
                    paper_bgcolor=theme_colors["paper"],
                    plot_bgcolor=theme_colors["background"],
                    font=dict(
                        family="Arial, sans-serif",
                        size=13,
                        color=chart_text,
                    ),
                    showlegend=False,
                    xaxis=dict(
                        title=dict(
                            text="Month",
                            font=dict(size=13, color=chart_text),
                        ),
                        type="category",
                        tickangle=0,
                        tickfont=dict(size=12, color=chart_muted),
                        automargin=True,
                        showgrid=False,
                        showline=True,
                        linecolor=chart_grid,
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=dict(
                            text="Spending (RM)",
                            font=dict(size=13, color=chart_text),
                        ),
                        tickfont=dict(size=12, color=chart_muted),
                        rangemode="tozero",
                        tickformat=",.2f",
                        gridcolor=chart_grid,
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor=chart_grid,
                        zerolinewidth=1.5,
                    ),
                    hoverlabel=dict(
                        bgcolor=theme_colors["paper"],
                        font=dict(color=chart_text, size=12),
                    ),
                )

                st.plotly_chart(
                    monthly_report_fig,
                    use_container_width=True,
                )

        if "expense_category" in expenses_df.columns and not expenses_df.empty:
            category_totals = expenses_df.groupby("expense_category")["total_amount"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Highest-Spending Category", str(category_totals.idxmax()))
            c2.metric("Average Expense", f"RM{expenses_df['total_amount'].mean():,.2f}")
            largest = expenses_df.loc[expenses_df["total_amount"].idxmax()]
            c3.metric("Largest Expense", f"RM{float(largest['total_amount']):,.2f}", str(largest.get("merchant_name", "")))

# --------------------------------------------------
# Budget
# --------------------------------------------------
elif page == "Budget":
    st.title("💳 Monthly Budget")
    st.caption(
        "Plan your monthly spending, track expenses against your limit, "
        "and monitor your remaining budget."
    )
    today = date.today()
    month_key = today.strftime("%Y-%m")
    saved_budget = get_monthly_budget(current_user_id, month_key)
    default_budget = float(saved_budget) if saved_budget is not None else 500.0
    month_spending = 0.0
    if not expenses_df.empty and "transaction_date" in expenses_df.columns:
        budget_df = expenses_df.dropna(subset=["transaction_date"]).copy()
        month_spending = float(budget_df.loc[budget_df["transaction_date"].dt.strftime("%Y-%m") == month_key, "total_amount"].sum())
    with st.form("monthly_budget_form"):
        budget_input = st.number_input("Monthly budget (RM)", min_value=1.0, value=max(default_budget, 1.0), step=50.0)
        save_budget = st.form_submit_button("Save Budget", type="primary")
    if save_budget:
        save_monthly_budget(current_user_id, month_key, float(budget_input))
        st.success("Monthly budget saved successfully.")
        st.rerun()
    active_budget = get_monthly_budget(current_user_id, month_key)
    if active_budget is not None and float(active_budget) > 0:
        active_budget = float(active_budget)
        remaining = active_budget - month_spending
        used = month_spending / active_budget
        b1, b2, b3 = st.columns(3)
        b1.metric("Monthly Budget", f"RM{active_budget:,.2f}")
        b2.metric("Spent This Month", f"RM{month_spending:,.2f}")
        b3.metric("Budget Remaining", f"RM{remaining:,.2f}")
        st.progress(min(max(used, 0.0), 1.0))
        st.caption(f"{used:.1%} of your budget used")
        
        if used >= 1.0:
            alert_key = "budget_alert_shown_this_login"

            alert_message = (
                f"You have exceeded your monthly budget by "
                f"RM{abs(remaining):,.2f}."
            )

            st.error(f"🚨 Budget exceeded! {alert_message}")

            if not st.session_state.get(alert_key, False):

                @st.dialog("🚨 Monthly Budget Exceeded")
                def show_budget_alert():
                    st.error(alert_message)
                    st.write(
                        f"Monthly budget: RM{active_budget:,.2f}"
                    )
                    st.write(
                        f"Total spent: RM{month_spending:,.2f}"
                    )
                    st.write(
                        f"Amount over budget: RM{abs(remaining):,.2f}"
                    )

                    if st.button(
                        "Got it",
                        key=f"dismiss_{alert_key}",
                    ):
                        st.session_state[alert_key] = True
                        st.rerun()

                show_budget_alert()

        elif used >= 0.7:
            alert_key = "budget_alert_shown_this_login"

            alert_message = (
                f"You have used {used:.1%} of your monthly budget. "
                f"Only RM{remaining:,.2f} remains."
            )

            st.error(f"⚠️ Budget warning! {alert_message}")

            if not st.session_state.get(alert_key, False):

                @st.dialog("⚠️ Monthly Budget Warning")
                def show_budget_warning():
                    st.error(alert_message)
                    st.write(
                        f"Monthly budget: RM{active_budget:,.2f}"
                    )
                    st.write(
                        f"Total spent: RM{month_spending:,.2f}"
                    )
                    st.write(
                        f"Budget remaining: RM{remaining:,.2f}"
                    )

                    if st.button(
                        "Got it",
                        key=f"dismiss_{alert_key}",
                    ):
                        st.session_state[alert_key] = True
                        st.rerun()

                show_budget_warning()

        else:
            st.success("✅ Your spending is within budget.")

