import streamlit as st
import sys
import os

# Add the CAG directories to the Python path
cag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(cag_path)

# Import CAG components
from cag.cache.cache_store import CacheStore, load_cache as seed_cache
from cag.cag_pipeline.query_handler import run_query as cag_run_query

# Set page configuration
st.set_page_config(
    page_title="CAG Query Interface",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for minimal styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_cag_components():
    """Initialize and cache the CAG components to avoid reloading"""
    try:
        cache = CacheStore()
        seed_cache(cache)
        return cache
    except Exception as e:
        st.error(f"Error initializing CAG components: {str(e)}")
        st.error(f"Make sure the cache directory exists and is accessible.")
        return None

def main():
    # Simple header
    st.markdown("<h1 class='main-header'>CAG Query Interface</h1>", unsafe_allow_html=True)
    
    # Load CAG components
    cache = initialize_cag_components()
    
    if cache is None:
        st.warning("Unable to load CAG components. Please check your configuration.")
        return
    
    # Query input section
    query = st.text_area("Enter your query:", height=100)
    
    # Simple configuration in a small expander
    with st.expander("Settings", expanded=False):
        relevance_threshold = st.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    # Submit button
    if st.button("Submit", type="primary"):
        if not query:
            st.warning("Please enter a query.")
            return
        
        with st.spinner("Generating response..."):
            try:
                # Generate CAG answer
                result = cag_run_query(
                    cache,
                    query,
                    debug=False,
                    relevance_threshold=relevance_threshold
                )
                
                # Display ONLY the model_response without any citations or references
                st.markdown("### Response")
                if isinstance(result, dict):  # If using the newer function version that returns a dict
                    # Only use the direct model response without citations
                    st.markdown(f"<div class='response-box'>{result['model_response']}</div>", unsafe_allow_html=True)
                else:  # If using the older function version that returns a string
                    # For string return, try to extract just the response part without citations
                    clean_response = result
                    # Remove references section if present
                    if "### References" in result:
                        clean_response = result.split("### References")[0]
                    # Remove citations if present (e.g., [1], [2], etc.)
                    if "\n\n[1]" in clean_response:
                        clean_response = clean_response.split("\n\n[1]")[0]
                    # Remove warning about no relevant documents if present
                    if clean_response.startswith("Note: No highly relevant documents were found"):
                        parts = clean_response.split("\n\n", 1)
                        if len(parts) > 1:
                            clean_response = parts[1]
                            
                    st.markdown(f"<div class='response-box'>{clean_response}</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")

if __name__ == "__main__":
    main()