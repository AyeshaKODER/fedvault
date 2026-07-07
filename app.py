"""FedVault Streamlit dashboard — federated learning control and explainability."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.worker import run_all_nodes, run_node
from config import (
    ACCENT_COLOR,
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    CENTRAL_SERVER_URL,
    CLASS_NAMES,
    CLIENT_CREDENTIALS,
    DARK_BG,
    DARK_SURFACE,
    GLOBAL_ROUNDS,
    NODE_DATA_PATHS,
    NODE_IDS,
)
from data_generator import generate_all_node_data
from ml.gradcam import run_gradcam_explanation
from ml.model import (
    apply_state_dict_to_model,
    count_dataset_distribution,
    get_resnet18,
    predict_single_image,
)

# ---------------------------------------------------------------------------
# Page configuration and theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FedVault — Federated Chest X-Ray Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {DARK_BG};
        color: #e8eaed;
    }}
    [data-testid="stSidebar"] {{
        background-color: {DARK_SURFACE};
        border-right: 1px solid {ACCENT_COLOR}33;
    }}
    h1, h2, h3, h4 {{
        color: {ACCENT_COLOR} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    .fedvault-metric-card {{
        background: linear-gradient(135deg, {DARK_SURFACE} 0%, #121826 100%);
        border: 1px solid {ACCENT_COLOR}44;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 0 18px {ACCENT_COLOR}22;
    }}
    .fedvault-metric-card h4 {{
        margin: 0 0 0.4rem 0;
        font-size: 0.85rem;
        color: #9aa0a6 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .fedvault-metric-card p {{
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    .fedvault-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: {ACCENT_COLOR}22;
        border: 1px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .stButton > button {{
        background: linear-gradient(90deg, {ACCENT_COLOR} 0%, #00b8d4 100%);
        color: #0e1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        box-shadow: 0 0 20px {ACCENT_COLOR}88;
        transform: translateY(-1px);
    }}
    div[data-testid="stMetricValue"] {{
        color: {ACCENT_COLOR};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #9aa0a6;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT_COLOR} !important;
        border-bottom: 2px solid {ACCENT_COLOR};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="fedvault-metric-card">
            <h4>{title}</h4>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_data_generated() -> None:
    try:
        generate_all_node_data(force=False)
    except Exception as exc:
        st.error(f"Failed to generate synthetic datasets: {exc}")


def get_admin_token() -> str | None:
    if "admin_token" in st.session_state and st.session_state.admin_token:
        return st.session_state.admin_token
    try:
        response = requests.post(
            f"{CENTRAL_SERVER_URL}/token",
            data={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
            timeout=15,
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        st.session_state.admin_token = token
        return token
    except Exception:
        return None


def server_is_online() -> bool:
    try:
        response = requests.get(f"{CENTRAL_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def fetch_admin_status() -> dict | None:
    token = get_admin_token()
    if not token:
        return None
    try:
        response = requests.get(
            f"{CENTRAL_SERVER_URL}/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_public_status() -> dict | None:
    try:
        response = requests.get(f"{CENTRAL_SERVER_URL}/status/public", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def start_global_round() -> dict | None:
    token = get_admin_token()
    if not token:
        st.error("Unable to authenticate as admin.")
        return None
    try:
        response = requests.post(
            f"{CENTRAL_SERVER_URL}/start-round",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Failed to start global round: {exc}")
        return None


def load_global_model_for_inference() -> torch.nn.Module | None:
    token = get_admin_token()
    if not token:
        return None
    try:
        response = requests.get(
            f"{CENTRAL_SERVER_URL}/global-model/admin",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        model = get_resnet18(pretrained=False)
        apply_state_dict_to_model(model, payload["weights"])
        model.eval()
        return model
    except Exception as exc:
        st.warning(f"Could not load global model from server: {exc}. Using local initialized weights.")
        model = get_resnet18(pretrained=True)
        model.eval()
        return model


def list_node_images(node_id: str, split: str = "test") -> list[Path]:
    node_root = NODE_DATA_PATHS[node_id]
    images: list[Path] = []
    for class_name in CLASS_NAMES:
        class_dir = node_root / split / class_name
        if class_dir.exists():
            images.extend(sorted(class_dir.glob("*.png")))
    return images


def render_loss_convergence_chart(round_history: list[dict]) -> None:
    if not round_history:
        st.info("No completed global rounds yet. Trigger a round to begin federated training.")
        return

    rounds = [entry["round"] for entry in round_history]
    losses = [entry["avg_train_loss"] for entry in round_history]
    accuracies = [entry["avg_accuracy"] * 100 for entry in round_history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=losses,
            mode="lines+markers",
            name="Avg Train Loss",
            line=dict(color=ACCENT_COLOR, width=3),
            marker=dict(size=10, color=ACCENT_COLOR),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=accuracies,
            mode="lines+markers",
            name="Avg Accuracy (%)",
            yaxis="y2",
            line=dict(color="#ff6b6b", width=3, dash="dot"),
            marker=dict(size=10, color="#ff6b6b"),
        )
    )
    fig.update_layout(
        title=dict(text="Global Federated Training Convergence", font=dict(color=ACCENT_COLOR, size=18)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_SURFACE,
        font=dict(color="#e8eaed"),
        xaxis=dict(title="Global Round", gridcolor="#2a3142", dtick=1),
        yaxis=dict(title="Average Train Loss", gridcolor="#2a3142"),
        yaxis2=dict(title="Average Accuracy (%)", overlaying="y", side="right", gridcolor="#2a3142"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaed")),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_admin_tab() -> None:
    st.header("Central Server — Admin Control")
    st.markdown(
        '<span class="fedvault-badge">FedAvg Orchestration</span>',
        unsafe_allow_html=True,
    )

    status = fetch_admin_status() or fetch_public_status()

    col1, col2, col3, col4 = st.columns(4)
    current_round = status["current_round"] if status else 0
    max_rounds = status["max_rounds"] if status else GLOBAL_ROUNDS
    round_active = status["round_active"] if status else False
    round_history = status.get("round_history", []) if status else []

    with col1:
        metric_card("Current Round", f"{current_round} / {max_rounds}")
    with col2:
        metric_card("Round Status", "ACTIVE" if round_active else "IDLE")
    with col3:
        completed = len(status.get("completed_nodes", [])) if status else 0
        metric_card("Nodes Reported", f"{completed} / {len(NODE_IDS)}")
    with col4:
        if round_history:
            latest = round_history[-1]
            metric_card("Latest Avg Accuracy", f"{latest['avg_accuracy'] * 100:.1f}%")
        else:
            metric_card("Latest Avg Accuracy", "N/A")

    st.divider()

    action_col1, action_col2 = st.columns([1, 2])
    with action_col1:
        if st.button("Trigger Global Round", type="primary", use_container_width=True):
            with st.spinner("Starting global round and dispatching hospital nodes..."):
                start_result = start_global_round()
                if start_result:
                    st.success(start_result["message"])
                    node_results = run_all_nodes()
                    st.session_state["last_node_results"] = node_results
                    time.sleep(1)
                    st.rerun()

    with action_col2:
        st.markdown(
            """
            **Orchestration flow:** Start round on central server → each hospital node downloads
            global weights → local PyTorch training → FedAvg aggregation when all nodes report.
            """
        )

    if "last_node_results" in st.session_state:
        with st.expander("Latest Node Execution Results", expanded=False):
            st.json(st.session_state["last_node_results"])

    st.subheader("Training Convergence")
    render_loss_convergence_chart(round_history)

    if round_history:
        st.subheader("Round History")
        st.dataframe(
            [
                {
                    "Round": entry["round"],
                    "Avg Loss": entry["avg_train_loss"],
                    "Avg Accuracy": f"{entry['avg_accuracy'] * 100:.2f}%",
                    "Nodes": ", ".join(entry["participating_nodes"]),
                    "Timestamp": entry["timestamp"],
                }
                for entry in round_history
            ],
            use_container_width=True,
        )


def render_client_tab() -> None:
    st.header("Hospital Node Monitor")
    status = fetch_public_status()

    for node_id in NODE_IDS:
        st.subheader(f"🏥 {node_id.replace('_', ' ').title()}")
        creds = CLIENT_CREDENTIALS[node_id]
        distribution = count_dataset_distribution(NODE_DATA_PATHS[node_id])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            metric_card("Client Username", creds["username"])
        with col_b:
            train_total = sum(distribution.get("train", {}).values())
            metric_card("Train Samples", str(train_total))
        with col_c:
            test_total = sum(distribution.get("test", {}).values())
            metric_card("Test Samples", str(test_total))

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Train Distribution**")
            st.json(distribution.get("train", {}))
        with dist_col2:
            st.markdown("**Test Distribution**")
            st.json(distribution.get("test", {}))

        node_logs = []
        if status:
            node_logs = [log for log in status.get("client_logs", []) if log.get("node_id") == node_id]

        if node_logs:
            st.markdown("**Weight Transmission Logs**")
            st.dataframe(node_logs, use_container_width=True)
        else:
            st.info("No transmission logs recorded for this node yet.")

        run_col1, run_col2 = st.columns([1, 3])
        with run_col1:
            if st.button(f"Run {node_id} Only", key=f"run_{node_id}"):
                if not server_is_online():
                    st.error("Central server is offline. Start the server first.")
                elif status and status.get("round_active"):
                    with st.spinner(f"Running federated round for {node_id}..."):
                        try:
                            result = run_node(node_id)
                            st.success(f"{node_id} completed successfully.")
                            st.json(result)
                        except Exception as exc:
                            st.error(f"{node_id} failed: {exc}")
                else:
                    st.warning("Start a global round from the Admin tab before running nodes.")

        st.divider()


def render_explainability_tab() -> None:
    st.header("Grad-CAM Explainability")
    st.markdown(
        "Select a synthetic chest X-ray from a hospital node, classify it with the current global model, "
        "and visualize Grad-CAM activations from ResNet-18 `layer4`."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_node = st.selectbox("Hospital Node", NODE_IDS)
    with col2:
        selected_split = st.selectbox("Dataset Split", ["test", "train"])
    with col3:
        images = list_node_images(selected_node, split=selected_split)
        if images:
            selected_image = st.selectbox(
                "X-Ray Image",
                images,
                format_func=lambda p: p.name,
            )
        else:
            selected_image = None
            st.warning("No images found for this node/split.")

    if selected_image is None:
        return

    if st.button("Run Classification + Grad-CAM", type="primary"):
        with st.spinner("Loading global model and generating explanation..."):
            try:
                model = load_global_model_for_inference()
                if model is None:
                    st.error("Model could not be loaded.")
                    return

                prediction = predict_single_image(model, selected_image)
                explanation = run_gradcam_explanation(
                    model,
                    prediction["input_tensor"],
                    target_class=prediction["predicted_index"],
                )

                st.session_state["explain_result"] = {
                    "prediction": prediction,
                    "explanation": explanation,
                    "image_path": str(selected_image),
                    "true_class": selected_image.parent.name,
                }
            except Exception as exc:
                st.error(f"Explainability pipeline failed: {exc}")
                return

    if "explain_result" not in st.session_state:
        return

    result = st.session_state["explain_result"]
    prediction = result["prediction"]
    explanation = result["explanation"]
    true_class = result["true_class"]
    predicted_class = prediction["predicted_class"]
    confidence = prediction["probabilities"][predicted_class] * 100

    pred_col1, pred_col2, pred_col3 = st.columns(3)
    with pred_col1:
        metric_card("True Label", true_class.title())
    with pred_col2:
        metric_card("Predicted Label", predicted_class.title())
    with pred_col3:
        metric_card("Confidence", f"{confidence:.1f}%")

    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.progress(prediction["probabilities"]["normal"], text=f"Normal: {prediction['probabilities']['normal'] * 100:.1f}%")
    with prob_col2:
        st.progress(
            prediction["probabilities"]["pneumonia"],
            text=f"Pneumonia: {prediction['probabilities']['pneumonia'] * 100:.1f}%",
        )

    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.markdown("**Original X-Ray**")
        st.image(explanation["original"], use_container_width=True)
    with img_col2:
        st.markdown("**Grad-CAM Heatmap**")
        cam_display = np.uint8(255 * explanation["cam"])
        st.image(cam_display, use_container_width=True, clamp=True)
    with img_col3:
        st.markdown("**Overlay**")
        st.image(explanation["overlay"], use_container_width=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/hospital.png",
            width=72,
        )
        st.title("FedVault")
        st.caption("Decentralized Federated Learning for Healthcare AI")

        st.divider()
        online = server_is_online()
        if online:
            st.success("Central Server: Online")
        else:
            st.error("Central Server: Offline")
            st.code(
                "uvicorn server.main:app --host 127.0.0.1 --port 8000",
                language="bash",
            )

        st.divider()
        st.markdown("**Research Prototype**")
        st.markdown(
            f"""
            - ResNet-18 (CPU)
            - FedAvg aggregation
            - JWT auth (Admin / Client)
            - Grad-CAM explainability
            - Accent: `{ACCENT_COLOR}`
            """
        )

        if st.button("Refresh Dashboard"):
            st.rerun()


def main() -> None:
    ensure_data_generated()

    st.title("🏛️ FedVault")
    st.markdown(
        f"<p style='color:#9aa0a6; font-size:1.1rem;'>Federated Chest X-Ray Classification — "
        f"<span style='color:{ACCENT_COLOR};'>Privacy-Preserving Healthcare AI</span></p>",
        unsafe_allow_html=True,
    )

    render_sidebar()

    tab_admin, tab_client, tab_explain = st.tabs(
        ["Admin Control", "Client Nodes", "Explainability"]
    )

    with tab_admin:
        if not server_is_online():
            st.warning(
                "The central server is not reachable. Start it with:\n\n"
                "`uvicorn server.main:app --host 127.0.0.1 --port 8000`"
            )
        render_admin_tab()

    with tab_client:
        render_client_tab()

    with tab_explain:
        render_explainability_tab()


if __name__ == "__main__":
    main()
