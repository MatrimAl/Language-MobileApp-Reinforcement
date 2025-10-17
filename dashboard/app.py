import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Configuration
st.set_page_config(
    page_title="RL Language Learning Dashboard",
    page_icon="🧠",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000/api"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 RL Language Learning Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Model Status
st.sidebar.header("📊 Model Status")

try:
    model_info_response = requests.get(f"{API_BASE_URL}/rl/model/info")
    model_info = model_info_response.json()
    
    if model_info["status"] == "loaded":
        st.sidebar.success("✅ Model Loaded")
        st.sidebar.metric("Epsilon", f"{model_info['epsilon']:.4f}")
        st.sidebar.metric("Memory Size", model_info['memory_size'])
        st.sidebar.metric("Training Episodes", model_info['training_episodes'])
    else:
        st.sidebar.warning("⚠️ Model Not Loaded")
        if st.sidebar.button("🚀 Initialize Sample Model"):
            with st.spinner("Initializing model..."):
                init_response = requests.post(f"{API_BASE_URL}/rl/initialize")
                if init_response.status_code == 200:
                    st.sidebar.success("Model initialized!")
                    st.rerun()
except:
    st.sidebar.error("❌ Backend Offline")

# Training Section
st.sidebar.header("🎓 Training")

if st.sidebar.button("▶️ Start Training"):
    with st.spinner("Starting training..."):
        train_response = requests.post(
            f"{API_BASE_URL}/rl/train",
            json={"episodes": 100, "batch_size": 32, "learning_rate": 0.001}
        )
        if train_response.status_code == 200:
            st.sidebar.success("Training started!")

# Training Status
try:
    training_status_response = requests.get(f"{API_BASE_URL}/rl/training/status")
    training_status = training_status_response.json()
    
    if training_status["is_training"]:
        st.sidebar.info("🔄 Training in progress...")
        progress = training_status["progress"]
        st.sidebar.progress(progress / 100)
        st.sidebar.text(f"Episode: {training_status['episode']}")
except:
    pass

# Main Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Metrics", "🎯 RL Visualization", "👥 User Analytics", "🧪 Testing"])

# TAB 1: Model Metrics
with tab1:
    st.header("📊 Training Metrics")
    
    try:
        metrics_response = requests.get(f"{API_BASE_URL}/rl/model/metrics")
        
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Episodes", metrics["total_episodes"])
            
            with col2:
                st.metric("Avg Reward", f"{metrics['avg_reward']:.2f}")
            
            with col3:
                st.metric("Recent Avg (100)", f"{metrics['recent_avg_reward']:.2f}")
            
            with col4:
                st.metric("Max Reward", f"{metrics['max_reward']:.2f}")
            
            # Episode Rewards Chart
            st.subheader("📈 Episode Rewards")
            
            rewards_df = pd.DataFrame({
                'Episode': range(len(metrics["episode_rewards"])),
                'Reward': metrics["episode_rewards"]
            })
            
            fig_rewards = go.Figure()
            fig_rewards.add_trace(go.Scatter(
                x=rewards_df['Episode'],
                y=rewards_df['Reward'],
                mode='lines',
                name='Episode Reward',
                line=dict(color='#667eea', width=2)
            ))
            
            # Moving average
            if len(rewards_df) > 10:
                rewards_df['MA'] = rewards_df['Reward'].rolling(window=10).mean()
                fig_rewards.add_trace(go.Scatter(
                    x=rewards_df['Episode'],
                    y=rewards_df['MA'],
                    mode='lines',
                    name='Moving Average (10)',
                    line=dict(color='#764ba2', width=3, dash='dash')
                ))
            
            fig_rewards.update_layout(
                title="Episode Rewards Over Time",
                xaxis_title="Episode",
                yaxis_title="Reward",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_rewards, use_container_width=True)
            
            # Epsilon Decay
            st.subheader("🎲 Exploration vs Exploitation")
            
            epsilon_df = pd.DataFrame({
                'Episode': range(len(metrics["epsilon_values"])),
                'Epsilon': metrics["epsilon_values"]
            })
            
            fig_epsilon = px.line(
                epsilon_df,
                x='Episode',
                y='Epsilon',
                title='Epsilon Decay (Exploration Rate)',
                template='plotly_white'
            )
            fig_epsilon.update_traces(line_color='#e74c3c', line_width=2)
            
            st.plotly_chart(fig_epsilon, use_container_width=True)
            
        else:
            st.warning("No training metrics available. Train the model first!")
    
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

# TAB 2: RL Visualization
with tab2:
    st.header("🎯 RL Agent Decision Visualization")
    
    st.subheader("State Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        level = st.slider("User Level", 1, 100, 10)
        total_learned = st.slider("Total Words Learned", 0, 1000, 50)
        avg_acc = st.slider("Average Accuracy", 0.0, 1.0, 0.7, 0.01)
        recent_acc = st.slider("Recent Accuracy", 0.0, 1.0, 0.8, 0.01)
        streak = st.slider("Current Streak", 0, 100, 5)
        time_hours = st.slider("Hours Since Last Session", 0.0, 168.0, 24.0, 1.0)
    
    with col2:
        mastery_0 = st.slider("Mastery 0-20%", 0.0, 1.0, 0.4, 0.01)
        mastery_1 = st.slider("Mastery 20-40%", 0.0, 1.0, 0.3, 0.01)
        mastery_2 = st.slider("Mastery 40-60%", 0.0, 1.0, 0.2, 0.01)
        mastery_3 = st.slider("Mastery 60-80%", 0.0, 1.0, 0.1, 0.01)
        mastery_4 = st.slider("Mastery 80-100%", 0.0, 1.0, 0.0, 0.01)
        avg_difficulty = st.slider("Avg Mastered Difficulty", 0, 5, 2)
    
    state = [
        level, total_learned, avg_acc, recent_acc, streak, time_hours,
        mastery_0, mastery_1, mastery_2, mastery_3, mastery_4, avg_difficulty
    ]
    
    if st.button("🔮 Predict Best Action", type="primary"):
        try:
            predict_response = requests.post(
                f"{API_BASE_URL}/rl/predict",
                json={"state": state}
            )
            
            if predict_response.status_code == 200:
                prediction = predict_response.json()
                
                st.success(f"🎯 Recommended Difficulty: **{prediction['difficulty']}**")
                st.info(f"🤖 Exploration Rate: {prediction['exploration_rate']:.3f}")
                
                # Q-Values Bar Chart
                st.subheader("Q-Values for Each Action")
                
                q_df = pd.DataFrame({
                    'Difficulty': ['Beginner', 'Elementary', 'Intermediate', 'Advanced', 'Expert'],
                    'Q-Value': prediction['q_values']
                })
                
                fig_q = px.bar(
                    q_df,
                    x='Difficulty',
                    y='Q-Value',
                    title='Q-Values by Difficulty Level',
                    color='Q-Value',
                    color_continuous_scale='Viridis'
                )
                
                fig_q.update_layout(template='plotly_white')
                st.plotly_chart(fig_q, use_container_width=True)
                
                # Confidence
                st.metric("Decision Confidence", f"{prediction['confidence']:.3f}")
                
        except Exception as e:
            st.error(f"Error predicting action: {e}")

# TAB 3: User Analytics
with tab3:
    st.header("👥 User Learning Analytics")
    
    st.info("🚧 User analytics will be available after user data collection")
    
    # Placeholder for future user analytics
    st.subheader("📊 Sample User Metrics")
    
    # Sample data
    sample_users = pd.DataFrame({
        'User': ['User 1', 'User 2', 'User 3', 'User 4', 'User 5'],
        'Words Learned': [45, 67, 23, 89, 34],
        'Accuracy': [0.75, 0.82, 0.68, 0.91, 0.73],
        'Level': [3, 4, 2, 6, 3]
    })
    
    fig_users = px.scatter(
        sample_users,
        x='Words Learned',
        y='Accuracy',
        size='Level',
        color='Level',
        hover_data=['User'],
        title='User Progress Overview',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_users, use_container_width=True)

# TAB 4: Testing
with tab4:
    st.header("🧪 API Testing")
    
    st.subheader("Health Check")
    if st.button("Check API Health"):
        try:
            health_response = requests.get(f"{API_BASE_URL.replace('/api', '')}/health")
            if health_response.status_code == 200:
                st.success("✅ API is healthy!")
                st.json(health_response.json())
            else:
                st.error("❌ API health check failed")
        except:
            st.error("❌ Cannot connect to API")
    
    st.subheader("Model Info")
    if st.button("Get Model Info"):
        try:
            model_response = requests.get(f"{API_BASE_URL}/rl/model/info")
            if model_response.status_code == 200:
                st.json(model_response.json())
        except:
            st.error("❌ Cannot get model info")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 <strong>reinFORCING_the_people</strong> - RL Language Learning Dashboard</p>
    <p>Powered by DQN & FastAPI | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh option
if st.sidebar.checkbox("🔄 Auto Refresh (5s)"):
    time.sleep(5)
    st.rerun()
