import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TEST_ITERATIONS = int(os.getenv("TEST_ITERATIONS", "3"))

MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]

REGIONS = [
    "global",
    "us-central1",
    "us-east5",
    "us-east1",
    "asia-northeast1",
    "asia-northeast3",
    "us-central4",
    "europe-west4",
    "us-east4",
    "europe-west1",
]

# TPU availability by region
TPU_INFO = {
    "us-central1": ["Trillium (v6e)", "TPU v5e"],
    "us-east5": ["Trillium (v6e)", "TPU v5p", "TPU v5e"],
    "us-east1": ["Trillium (v6e)", "TPU v5p"],
    "asia-northeast1": ["Trillium (v6e)"],
    "europe-west4": ["Trillium (v6e)", "TPU v5p", "TPU v5e"],
    "europe-west1": ["TPU v5e"],
}

# Test prompt - meaningful weight for accurate latency measurement
TEST_PROMPT = """You are a senior software architect. Please analyze the following scenario and provide a detailed response.

Scenario: A startup is building a real-time collaborative document editing platform similar to Google Docs. They expect to handle 100,000 concurrent users within the first year. The platform needs to support:
1. Real-time text synchronization across multiple clients
2. Offline editing with conflict resolution
3. Version history and rollback capabilities
4. Rich text formatting and embedded media
5. Comments and suggestions workflow

Please provide:
1. A high-level architecture recommendation (2-3 paragraphs)
2. Key technology choices with brief justifications
3. Top 3 potential scalability challenges and mitigation strategies

Keep your response concise but comprehensive."""


@dataclass
class TestResult:
    model: str
    region: str
    latency_ms: float
    tokens_generated: int
    success: bool
    error: str | None = None
    tpu_info: str | None = None


def get_tpu_display(region: str) -> str:
    """Get TPU information for display."""
    if region in TPU_INFO:
        return ", ".join(TPU_INFO[region])
    return "-"


def test_model_region(model: str, region: str) -> TestResult:
    """Test a specific model in a specific region."""
    tpu_info = get_tpu_display(region)

    try:
        # Create client for the specific region
        if region == "global":
            client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")
        else:
            client = genai.Client(vertexai=True, project=PROJECT_ID, location=region)

        # Measure response time
        start_time = time.perf_counter()

        response = client.models.generate_content(
            model=model,
            contents=TEST_PROMPT,
            config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Get token count if available
        tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return TestResult(
            model=model,
            region=region,
            latency_ms=latency_ms,
            tokens_generated=tokens,
            success=True,
            tpu_info=tpu_info,
        )

    except Exception as e:
        return TestResult(
            model=model,
            region=region,
            latency_ms=0,
            tokens_generated=0,
            success=False,
            error=str(e),
            tpu_info=tpu_info,
        )


def run_tests(models: list[str], regions: list[str], iterations: int, progress_callback=None) -> list[TestResult]:
    """Run tests for all model-region combinations."""
    results = []
    total_tests = len(models) * len(regions) * iterations
    completed = 0

    for model in models:
        for region in regions:
            region_results = []
            for i in range(iterations):
                result = test_model_region(model, region)
                region_results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_tests, f"Testing {model} in {region} (iteration {i+1}/{iterations})")

            # Calculate average for successful tests
            successful_results = [r for r in region_results if r.success]
            if successful_results:
                avg_latency = statistics.mean([r.latency_ms for r in successful_results])
                avg_tokens = int(statistics.mean([r.tokens_generated for r in successful_results]))
                results.append(TestResult(
                    model=model,
                    region=region,
                    latency_ms=avg_latency,
                    tokens_generated=avg_tokens,
                    success=True,
                    tpu_info=successful_results[0].tpu_info,
                ))
            else:
                # All iterations failed
                results.append(region_results[0])

    return results


def create_results_dataframe(results: list[TestResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    data = []
    for r in results:
        data.append({
            "Model": r.model,
            "Region": r.region,
            "Latency (ms)": round(r.latency_ms, 2) if r.success else "N/A",
            "Tokens": r.tokens_generated if r.success else "N/A",
            "TPU": r.tpu_info or "-",
            "Status": "Success" if r.success else f"Error: {r.error}",
        })
    return pd.DataFrame(data)


def create_latency_chart(results: list[TestResult]) -> go.Figure:
    """Create a grouped bar chart for latency comparison."""
    successful = [r for r in results if r.success]

    df = pd.DataFrame([
        {"Model": r.model, "Region": r.region, "Latency (ms)": r.latency_ms, "TPU": r.tpu_info}
        for r in successful
    ])

    fig = px.bar(
        df,
        x="Region",
        y="Latency (ms)",
        color="Model",
        barmode="group",
        title="Response Latency by Model and Region",
        hover_data=["TPU"],
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )

    return fig


def create_heatmap(results: list[TestResult]) -> go.Figure:
    """Create a heatmap of latencies."""
    successful = [r for r in results if r.success]

    # Pivot data for heatmap
    models = list(set(r.model for r in successful))
    regions = list(set(r.region for r in successful))

    z_data = []
    for model in models:
        row = []
        for region in regions:
            matching = [r for r in successful if r.model == model and r.region == region]
            if matching:
                row.append(matching[0].latency_ms)
            else:
                row.append(None)
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=regions,
        y=models,
        colorscale="RdYlGn_r",
        text=[[f"{v:.0f}ms" if v else "N/A" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
    ))

    fig.update_layout(
        title="Latency Heatmap (ms) - Lower is Better",
        xaxis_tickangle=-45,
        height=400,
    )

    return fig


# Streamlit UI
st.set_page_config(
    page_title="Gemini Response Latency Test",
    page_icon="🚀",
    layout="wide",
)

st.title("Gemini Model Response Latency Test")
st.markdown("Compare response latency across different Gemini models and Google Cloud regions.")

# Sidebar configuration
st.sidebar.header("Configuration")

if not PROJECT_ID or PROJECT_ID == "your-project-id":
    st.sidebar.error("Please set PROJECT_ID in .env file")
    st.stop()

st.sidebar.success(f"Project: {PROJECT_ID}")

selected_models = st.sidebar.multiselect(
    "Select Models",
    MODELS,
    default=MODELS,
)

selected_regions = st.sidebar.multiselect(
    "Select Regions",
    REGIONS,
    default=REGIONS,
)

iterations = st.sidebar.slider(
    "Test Iterations (for averaging)",
    min_value=1,
    max_value=10,
    value=TEST_ITERATIONS,
)

# TPU Reference
st.sidebar.markdown("---")
st.sidebar.subheader("TPU Reference")
for region, tpus in TPU_INFO.items():
    st.sidebar.markdown(f"**{region}**: {', '.join(tpus)}")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Test Prompt")
    with st.expander("View test prompt", expanded=False):
        st.text(TEST_PROMPT)

with col2:
    run_button = st.button("🚀 Run Tests", type="primary", use_container_width=True)

if run_button:
    if not selected_models or not selected_regions:
        st.error("Please select at least one model and one region.")
    else:
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress: float, status: str):
            progress_bar.progress(progress)
            status_text.text(status)

        with st.spinner("Running tests..."):
            results = run_tests(selected_models, selected_regions, iterations, update_progress)

        progress_bar.progress(1.0)
        status_text.text("Tests completed!")

        # Store results in session state
        st.session_state.results = results

# Display results if available
if "results" in st.session_state and st.session_state.results:
    results = st.session_state.results

    st.markdown("---")
    st.header("Results")

    # Summary metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", len(results))
    with col2:
        st.metric("Successful", len(successful))
    with col3:
        st.metric("Failed", len(failed))
    with col4:
        if successful:
            avg_latency = statistics.mean([r.latency_ms for r in successful])
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")

    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🗺️ Heatmap", "📋 Data Table"])

    with tab1:
        if successful:
            fig = create_latency_chart(results)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if successful:
            fig = create_heatmap(results)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df = create_results_dataframe(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="gemini_latency_results.csv",
            mime="text/csv",
        )

    # Best performers
    if successful:
        st.markdown("---")
        st.subheader("🏆 Best Performers")

        for model in set(r.model for r in successful):
            model_results = [r for r in successful if r.model == model]
            if model_results:
                best = min(model_results, key=lambda x: x.latency_ms)
                tpu_badge = f" 🔥 **TPU: {best.tpu_info}**" if best.tpu_info and best.tpu_info != "-" else ""
                st.markdown(f"**{model}**: {best.region} ({best.latency_ms:.0f}ms){tpu_badge}")
