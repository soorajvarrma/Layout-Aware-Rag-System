"""
Streamlit UI for Layout-Aware RAG System
"""

import streamlit as st
from src.pipeline_v2 import ImprovedQAPipeline
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="BERT Paper QA System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS - FIXED
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f1f1f;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: #1f1f1f;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2ecc71;
        color: #1f1f1f;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metric-box {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    with st.spinner("Loading QA system..."):
        st.session_state.pipeline = ImprovedQAPipeline()
        st.session_state.history = []

# Header
st.markdown('<div class="main-header">BERT Paper QA System</div>', unsafe_allow_html=True)
st.markdown("**Layout-Aware Retrieval-Augmented Generation for Academic Papers**")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This system demonstrates high-precision question answering over the BERT paper 
    (Devlin et al., 2019) using:
    
    - Layout-aware PDF parsing
    - Hybrid semantic retrieval
    - LLM-based answer generation
    """)
    
    st.header("Configuration")
    top_k = st.slider("Number of sources to retrieve", 3, 10, 7)
    show_scores = st.checkbox("Show relevance scores", value=True)
    show_metadata = st.checkbox("Show chunk metadata", value=False)
    
    st.header("Example Questions")
    example_questions = [
        "What are the two pre-training tasks used in BERT?",
        "What datasets were used for pre-training BERT?",
        "How does BERT compare to GPT on GLUE tasks?",
        "What is the architecture of BERT's input representation?",
        "What learning rate is used for fine-tuning BERT?"
    ]
    
    for i, example in enumerate(example_questions):
        if st.button(f"{i+1}: {example}", key=f"example_{i}", use_container_width=True):
            st.session_state.current_query = example

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")
    
    # Query input
    query = st.text_area(
        "Enter your question about the BERT paper:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="e.g., What are the pre-training tasks in BERT?"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.subheader("Statistics")
    if st.session_state.history:
        st.metric("Questions Asked", len(st.session_state.history))
        last_response = st.session_state.history[-1]
        st.metric("Sources Retrieved", last_response['num_sources'])
    else:
        st.metric("Questions Asked", 0)
        st.metric("Sources Retrieved", 0)

# Clear functionality
if clear_button:
    st.session_state.current_query = ""
    st.rerun()

# Process query
if search_button and query.strip():
    with st.spinner("Processing your question..."):
        try:
            # Get answer
            response = st.session_state.pipeline.ask(query, top_k=top_k)
            
            # Add to history
            st.session_state.history.append(response)
            
            # Display answer - FIXED VERSION
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-box">{response["answer"]}</div>', 
                unsafe_allow_html=True
            )
            
            # Display sources
            st.markdown("### Retrieved Sources")
            st.caption(f"Showing {response['num_sources']} most relevant sources")
            
            # Create tabs for sources
            source_tabs = st.tabs([f"Source {i+1}" for i in range(len(response['sources']))])
            
            for i, (tab, source) in enumerate(zip(source_tabs, response['sources'])):
                with tab:
                    # Header with metadata
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.markdown(f"**Page {source['page']}** • Type: `{source['type']}`")
                    with col_b:
                        if show_scores:
                            st.metric("Relevance", f"{source['score']:.3f}")
                    with col_c:
                        if show_metadata:
                            st.caption(f"ID: {source['chunk_id']}")
                    
                    # Source text - FIXED VERSION
                    st.markdown(
                        f'<div class="source-box">{source["text"]}</div>', 
                        unsafe_allow_html=True
                    )
            
            # Expandable: Full response JSON
            with st.expander("View Full Response JSON"):
                st.json(response)
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.exception(e)

# Query history
if st.session_state.history:
    with st.expander("📜 Query History"):
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**Q{len(st.session_state.history)-i}:** {item['query']}")
            st.caption(f"Retrieved {item['num_sources']} sources")
            st.markdown("---")

# Footer
st.markdown("---")
st.caption("""
**Document:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)  
**System:** Layout-Aware RAG with Hybrid Retrieval  
**Built for:** Revin Techno Solutions Assessment
""")
