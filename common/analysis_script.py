import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime

def load_data(cag_file, rag_file, comparison_file=None):
    """Load data from CSV files"""
    cag_df = pd.read_csv(cag_file)
    rag_df = pd.read_csv(rag_file)
    
    comparison_df = None
    if comparison_file and os.path.exists(comparison_file):
        comparison_df = pd.read_csv(comparison_file)
    
    return cag_df, rag_df, comparison_df

def convert_time_to_seconds(time_str):
    """Convert time string to float seconds"""
    if not isinstance(time_str, str):
        return time_str
    return float(time_str.replace('s', ''))

def analyze_results(cag_df, rag_df, comparison_df, output_dir='.'):
    """Analyze benchmark results and generate visualizations"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing benchmark results...")
    
    # 1. Processing Time Comparison
    if comparison_df is not None:
        comparison_df['cag_time_sec'] = comparison_df['cag_time'].apply(convert_time_to_seconds)
        comparison_df['rag_time_sec'] = comparison_df['rag_time'].apply(convert_time_to_seconds)
        
        plt.figure(figsize=(8, 6))
        
        # Bar plot for processing times
        ax = plt.subplot(111)
        bar_width = 0.30
        indices = range(len(comparison_df))
        
        cag_bars = ax.bar([i - bar_width/2 for i in indices], 
                         comparison_df['cag_time_sec'], 
                         bar_width, 
                         label='RAG 1', 
                         color='cornflowerblue', 
                         alpha=0.7)
                         
        rag_bars = ax.bar([i + bar_width/2 for i in indices], 
                         comparison_df['rag_time_sec'], 
                         bar_width, 
                         label='RAG 2', 
                         color='indianred', 
                         alpha=0.7)
        
        ax.set_xlabel('Query')
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('NVD -- RAG 1 vs RAG 2 Processing Time')
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Q{i+1}" for i in indices], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        time_plot_path = os.path.join(output_dir, f'processing_time_comparison_{timestamp}.png')
        plt.savefig(time_plot_path)
        print(f"Processing time comparison saved to: {time_plot_path}")
        
        # 2. Document Count Comparison
        plt.figure(figsize=(8, 6))
        
        # Bar plot for document counts
        ax = plt.subplot(111)
        
        cag_doc_bars = ax.bar([i - bar_width/2 for i in indices], 
                             comparison_df['cag_doc_count'], 
                             bar_width, 
                             label='RAG 1', 
                             color='lightseagreen', 
                             alpha=0.7)
                             
        rag_doc_bars = ax.bar([i + bar_width/2 for i in indices], 
                             comparison_df['rag_doc_count'], 
                             bar_width, 
                             label='RAG 2', 
                             color='orange', 
                             alpha=0.7)
        
        ax.set_xlabel('Query')
        ax.set_ylabel('Number of Documents Used')
        ax.set_title('NVD -- RAG 1 vs RAG 2 Document Usage')
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Q{i+1}" for i in indices], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        doc_plot_path = os.path.join(output_dir, f'document_count_comparison_{timestamp}.png')
        plt.savefig(doc_plot_path)
        print(f"Document count comparison saved to: {doc_plot_path}")
        
        # 3. Response Length Comparison
        plt.figure(figsize=(8, 6))
        
        # Bar plot for response lengths
        ax = plt.subplot(111)
        
        cag_len_bars = ax.bar([i - bar_width/2 for i in indices], 
                             comparison_df['cag_response_length'], 
                             bar_width, 
                             label='RAG 1', 
                             color='royalblue', 
                             alpha=0.7)
                             
        rag_len_bars = ax.bar([i + bar_width/2 for i in indices], 
                             comparison_df['rag_response_length'], 
                             bar_width, 
                             label='RAG 2', 
                             color='gold', 
                             alpha=0.7)
        
        ax.set_xlabel('Query')
        ax.set_ylabel('Response Length (characters)')
        ax.set_title('NVD -- RAG 1 vs RAG 2 Response Length')
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Q{i+1}" for i in indices], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        len_plot_path = os.path.join(output_dir, f'response_length_comparison_{timestamp}.png')
        plt.savefig(len_plot_path)
        print(f"Response length comparison saved to: {len_plot_path}")
        
        # 4. NEW: Response Score Comparison
        plt.figure(figsize=(8, 6))
        
        # Bar plot for response scores
        ax = plt.subplot(111)
        
        # Check if comparison_df has these columns, otherwise get them from cag_df and rag_df
        if 'cag_response_score' in comparison_df.columns and 'rag_response_score' in comparison_df.columns:
            cag_scores = comparison_df['cag_response_score']
            rag_scores = comparison_df['rag_response_score']
        else:
            # Assuming queries are in the same order in all dataframes
            cag_scores = []
            rag_scores = []
            for query in comparison_df['query'] if 'query' in comparison_df.columns else range(len(indices)):
                cag_query_df = cag_df[cag_df['query'] == query] if 'query' in cag_df.columns else cag_df.iloc[query:query+1]
                rag_query_df = rag_df[rag_df['query'] == query] if 'query' in rag_df.columns else rag_df.iloc[query:query+1]
                
                cag_scores.append(float(cag_query_df['response_score'].iloc[0]) if 'response_score' in cag_df.columns else 0)
                rag_scores.append(float(rag_query_df['response_score'].iloc[0]) if 'response_score' in rag_df.columns else 0)
        
        cag_score_bars = ax.bar([i - bar_width/2 for i in indices], 
                              cag_scores, 
                              bar_width, 
                              label='RAG 1', 
                              color='mediumseagreen', 
                              alpha=0.7)
                              
        rag_score_bars = ax.bar([i + bar_width/2 for i in indices], 
                              rag_scores, 
                              bar_width, 
                              label='RAG 2', 
                              color='mediumpurple', 
                              alpha=0.7)
        
        ax.set_xlabel('Query')
        ax.set_ylabel('Response Score (0-4)')
        ax.set_title('NVD -- RAG 1 vs RAG 2 Response Scores')
        ax.set_xticks(indices)
        ax.set_xticklabels([f"Q{i+1}" for i in indices], rotation=45)
        ax.set_ylim(0, 4.5)  # Setting y-axis limit based on 0-4 score range
        ax.legend()
        
        plt.tight_layout()
        score_plot_path = os.path.join(output_dir, f'response_score_comparison_{timestamp}.png')
        plt.savefig(score_plot_path)
        print(f"Response score comparison saved to: {score_plot_path}")
        
    # 5. Summary Statistics
    # Calculate average response scores
    cag_avg_score = cag_df['response_score'].mean() if 'response_score' in cag_df.columns else 'N/A'
    rag_avg_score = rag_df['response_score'].mean() if 'response_score' in rag_df.columns else 'N/A'
    
    summary = {
        'RAG 1 Average Processing Time': cag_df['processing_time'].apply(convert_time_to_seconds).mean(),
        'RAG 2 Average Processing Time': rag_df['processing_time'].apply(convert_time_to_seconds).mean(),
        'RAG 1 Total Processing Time': cag_df['processing_time'].apply(convert_time_to_seconds).sum(),
        'RAG 2 Total Processing Time': rag_df['processing_time'].apply(convert_time_to_seconds).sum(),
        'RAG 1 Average Response Score': cag_avg_score,
        'RAG 2 Average Response Score': rag_avg_score,
    }
    
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if 'Score' in key:
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}s")
    
    # Create summary dataframe and save to CSV
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f'benchmark_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # 6. Create detailed report with queries
    report_path = os.path.join(output_dir, f'detailed_analysis_{timestamp}.html')
    
    # Using .format() instead of f-strings for the HTML template to avoid f-string backslash issues
    html_content = """
    <html>
    <head>
        <title>RAG 1 vs RAG 2 Benchmark Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            .query-section {{ margin-top: 30px; border-left: 5px solid #3498db; padding-left: 15px; }}
            .response {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }}
            .document {{ background-color: #f0f0f0; padding: 10px; margin-top: 5px; border-radius: 5px; font-size: 0.9em; }}
            .stats {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>RAG 1 vs RAG 2 Benchmark Analysis</h1>
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p>RAG 1 Average Processing Time: {:.2f}s</p>
            <p>RAG 2 Average Processing Time: {:.2f}s</p>
            <p>RAG 1 Total Processing Time: {:.2f}s</p>
            <p>RAG 2 Total Processing Time: {:.2f}s</p>
            <p>RAG 1 Average Response Score: {}</p>
            <p>RAG 2 Average Response Score: {}</p>
        </div>
    """.format(
        summary['RAG 1 Average Processing Time'],
        summary['RAG 2 Average Processing Time'],
        summary['RAG 1 Total Processing Time'],
        summary['RAG 2 Total Processing Time'],
        f"{summary['RAG 1 Average Response Score']:.2f}" if isinstance(summary['RAG 1 Average Response Score'], (int, float)) else summary['RAG 1 Average Response Score'],
        f"{summary['RAG 2 Average Response Score']:.2f}" if isinstance(summary['RAG 2 Average Response Score'], (int, float)) else summary['RAG 2 Average Response Score']
    )
    
    # Add per-query analysis
    for i, query in enumerate(cag_df['query'].unique()):
        try:
            cag_row = cag_df[cag_df['query'] == query].iloc[0]
            rag_row = rag_df[rag_df['query'] == query].iloc[0]
            
            cag_time = convert_time_to_seconds(cag_row['processing_time'])
            rag_time = convert_time_to_seconds(rag_row['processing_time'])
            
            # Fix for float issue - convert to string if needed
            cag_response = str(cag_row['response'])
            rag_response = str(rag_row['response'])
            cag_docs = str(cag_row['documents']).replace('\n', '<br>')
            rag_docs = str(rag_row['documents']).replace('\n', '<br>')
            
            # Get response scores if available
            cag_score = cag_row['response_score'] if 'response_score' in cag_row else 'N/A'
            rag_score = rag_row['response_score'] if 'response_score' in rag_row else 'N/A'
            
            # Using format() method instead of f-strings to avoid backslash issues
            query_html = """
            <div class="query-section">
                <h2>Query {0}: {1}</h2>
                
                <h3>Processing Times</h3>
                <p>RAG 1: {2:.2f}s | RAG 2: {3:.2f}s | Difference: {4:.2f}s</p>
                
                <h3>Response Scores</h3>
                <p>RAG 1: {11} | RAG 2: {12}</p>
                
                <h3>RAG 1 Response</h3>
                <div class="response">{5}</div>
                <p class="stats">Length: {6} characters</p>
                
                <h3>RAG 2 Response</h3>
                <div class="response">{7}</div>
                <p class="stats">Length: {8} characters</p>
                
                <h3>Documents Used</h3>
                <h4>RAG 1 Documents</h4>
                <div class="document">{9}</div>
                
                <h4>RAG 2 Documents</h4>
                <div class="document">{10}</div>
            </div>
            """.format(
                i+1,
                query,
                cag_time,
                rag_time,
                rag_time - cag_time,
                cag_response,
                len(cag_response),
                rag_response,
                len(rag_response),
                cag_docs,
                rag_docs,
                cag_score,
                rag_score
            )
            
            html_content += query_html
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
    
    html_content += """
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Detailed HTML report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze RAG 1 and RAG 2 benchmark results')
    parser.add_argument('--cag', required=True, help='Path to RAG 1 benchmark CSV')
    parser.add_argument('--rag', required=True, help='Path to RAG 2 benchmark CSV')
    parser.add_argument('--comparison', help='Path to comparison CSV (optional)')
    parser.add_argument('--output', default='benchmark_analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    cag_df, rag_df, comparison_df = load_data(args.cag, args.rag, args.comparison)
    analyze_results(cag_df, rag_df, comparison_df, args.output)