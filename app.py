import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Set wide layout for better page utilization
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# Custom CSS for animations and styling
st.markdown("""
<style>
    /* Root color variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-color: #00d4ff;
        --dark-bg: #0f1419;
        --card-bg: #1a1f2e;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #16213e 100%);
        border-right: 2px solid #667eea;
    }
    
    /* Smooth scroll animation */
    html {
        scroll-behavior: smooth;
    }
    
    /* Header animations */
    .header-main {
        animation: slideInDown 0.8s ease-out;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .subheader-main {
        animation: slideInUp 0.8s ease-out 0.2s both;
    }
    
    .description-text {
        animation: fadeIn 1s ease-out 0.4s both;
    }
    
    /* Keyframe animations */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    @keyframes glow {
        0%, 100% {
            text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        }
        50% {
            text-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    /* Button animations */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File uploader styling */
    .stFileUploader {
        animation: slideInUp 0.6s ease-out;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        background: rgba(102, 126, 234, 0.1) !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.15) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Slider styling */
    .stSlider {
        animation: fadeIn 0.6s ease-out 0.3s both;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Subheadings */
    .subheader-custom {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 20px 0 10px 0;
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
        border-left-color: #00d4ff !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Sidebar headings */
    .sidebar-heading {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 15px;
        animation: glow 2s ease-in-out infinite;
    }
    
    .sidebar-content {
        font-size: 14px;
        line-height: 1.8;
        color: #c0c0c0;
        animation: fadeIn 0.8s ease-out 0.2s both;
    }
    
    /* About section */
    .about-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        animation: slideInUp 0.6s ease-out 0.4s both;
    }
    
    /* Footer */
    .footer-custom {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-top: 2px solid #667eea;
        padding: 20px;
        text-align: center;
        margin-top: 40px;
        border-radius: 8px;
        animation: fadeIn 1s ease-out 0.6s both;
    }
    
    /* Warning and info boxes */
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(245, 87, 108, 0.05) 100%) !important;
        border-left: 4px solid #f5576c !important;
        border-radius: 8px !important;
    }
    
    /* Image containers */
    .image-container {
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 10px;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .image-container:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        transform: translateY(-5px);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f1419;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header with student info
st.markdown("""
    <h1 class='header-main' style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 48px;'>✨ Shape & Contour Analyzer</h1>
    <h3 class='subheader-main' style='text-align: center; color: #764ba2; letter-spacing: 1px;'>Developed by Raghav Karthik (22MIA1124)</h3>
    <p class='description-text' style='text-align: center; color: #a0a0a0; font-size: 16px; margin-bottom: 30px;'>🎨 A Computer Vision Project for Geometric Shape Detection, Counting, and Analysis</p>
""", unsafe_allow_html=True)

# Sidebar for instructions and controls
with st.sidebar:
    st.markdown("""
    <div class='sidebar-heading'>📋 Instructions</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-content'>
    • Upload an image with clear geometric shapes (e.g., on a white background)<br>
    • The app detects shapes like triangles, squares, rectangles, circles, etc.<br>
    • It counts objects, computes area/perimeter, and provides insights<br>
    • Best for binary/high-contrast images. Adjust threshold if needed<br>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Add a slider for threshold adjustment to improve detection
    st.markdown("""
    <div style='margin: 20px 0 10px 0;'>
        <p style='font-weight: 700; color: #667eea; font-size: 16px;'>🎛️ Threshold Control</p>
    </div>
    """, unsafe_allow_html=True)
    thresh_value = st.slider("Adjust sensitivity", min_value=50, max_value=200, value=127, help="Higher values detect darker shapes")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-heading'>ℹ️ About</div>
    <div class='about-section'>
        This app uses <strong>OpenCV</strong> for contour detection and feature extraction. 
        Built with <strong>Streamlit</strong> for an interactive experience.
        <br><br>
        <span style='color: #00d4ff; font-weight: 600;'>✓ Real-time Shape Detection</span><br>
        <span style='color: #00d4ff; font-weight: 600;'>✓ Geometric Analysis</span><br>
        <span style='color: #00d4ff; font-weight: 600;'>✓ Visual Insights</span>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
<div class='subheader-custom'>📤 Upload Your Image</div>
""", unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an image (JPG, PNG, BMP)...", type=["jpg", "png", "bmp"])

if uploaded_image is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding with user-adjustable value
    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count objects (filter small noise)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    object_count = len(contours)
    
    # Prepare results list
    results = []
    output_img = img.copy()
    shape_types = {}  # For insights
    
    for i, cnt in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Approximate polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        sides = len(approx)
        
        # Determine shape
        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif sides == 5:
            shape = "Pentagon"
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            shape = "Circle" if circularity > 0.85 else "Ellipse/Unknown"
        
        # Track shape types for insights
        shape_types[shape] = shape_types.get(shape, 0) + 1
        
        results.append({
            "Object ID": i + 1,
            "Shape": shape,
            "Area (pixels)": round(area, 2),
            "Perimeter (pixels)": round(perimeter, 2)
        })
        
        # Draw contours and labels
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output_img, f"{shape} {i+1}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display results in two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='subheader-custom'>🖼️ Original Image</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='subheader-custom'>✨ Processed Image with Contours</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(output_img, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Object count and metrics table
    st.markdown(f"""
    <div class='subheader-custom'>📊 Analysis Results (Total Objects: <span style='color: #00d4ff;'>{object_count}</span>)</div>
    """, unsafe_allow_html=True)
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df.style.highlight_max(axis=0, color='#667eea').format(precision=2), use_container_width=True)
    else:
        st.markdown("""
        <div class='stWarning'>
            <p style='margin: 0; color: #f5576c;'>⚠️ No significant shapes detected. Try adjusting the threshold or uploading a clearer image.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Insights and Inferences Section
    if results:
        st.markdown("""
        <div class='subheader-custom'>💡 Insights and Inferences</div>
        """, unsafe_allow_html=True)
        with st.expander("📈 View Detailed Insights", expanded=True):
            # Average metrics
            avg_area = np.mean([r['Area (pixels)'] for r in results])
            avg_perim = np.mean([r['Perimeter (pixels)'] for r in results])
            largest_shape = max(results, key=lambda x: x['Area (pixels)'])
            smallest_shape = min(results, key=lambda x: x['Area (pixels)'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #667eea; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='margin: 5px 0; color: #667eea; font-weight: 700;'>📐 Average Area</p>
                    <p style='margin: 5px 0; color: #00d4ff; font-size: 20px; font-weight: 600;'>{round(avg_area, 2)} px</p>
                    <p style='margin: 5px 0; font-size: 12px; color: #a0a0a0;'>Indicates typical size of shapes</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #764ba2; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='margin: 5px 0; color: #764ba2; font-weight: 700;'>🔄 Average Perimeter</p>
                    <p style='margin: 5px 0; color: #00d4ff; font-size: 20px; font-weight: 600;'>{round(avg_perim, 2)} px</p>
                    <p style='margin: 5px 0; font-size: 12px; color: #a0a0a0;'>Complexity of boundaries</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='margin: 5px 0; color: #00d4ff; font-weight: 700;'>🎯 Largest Shape</p>
                    <p style='margin: 5px 0; color: #f5576c; font-size: 18px; font-weight: 600;'>{largest_shape['Shape']} #{largest_shape['Object ID']}</p>
                    <p style='margin: 5px 0; font-size: 12px; color: #a0a0a0;'>Area: {largest_shape['Area (pixels)']} px</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(245, 159, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%); border-left: 4px solid #f093fb; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <p style='margin: 5px 0; color: #f093fb; font-weight: 700;'>📍 Smallest Shape</p>
                    <p style='margin: 5px 0; color: #f5576c; font-size: 18px; font-weight: 600;'>{smallest_shape['Shape']} #{smallest_shape['Object ID']}</p>
                    <p style='margin: 5px 0; font-size: 12px; color: #a0a0a0;'>Area: {smallest_shape['Area (pixels)']} px</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Multiple visualization charts
            if shape_types:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class='subheader-custom'>📊 Data Visualizations</div>
                """, unsafe_allow_html=True)
                
                # Create a 2x2 grid for charts
                chart_col1, chart_col2 = st.columns(2)
                
                # 1. Pie Chart - Shape Distribution
                with chart_col1:
                    st.markdown("""
                    <p style='color: #667eea; font-weight: 700; font-size: 16px; margin-bottom: 10px;'>🥧 Shape Distribution</p>
                    """, unsafe_allow_html=True)
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    colors = ['#667eea', '#764ba2', '#00d4ff', '#f093fb', '#f5576c', '#ffa726', '#26c6da', '#ab47bc']
                    wedges, texts, autotexts = ax1.pie(
                        shape_types.values(), 
                        labels=shape_types.keys(), 
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=colors[:len(shape_types)],
                        textprops={'fontfamily': 'monospace', 'weight': 'bold', 'size': 11}
                    )
                    # Style the text with better font
                    for text in texts:
                        text.set_color('#ffffff')
                        text.set_fontsize(13)
                        text.set_weight('bold')
                        text.set_family('monospace')
                    for autotext in autotexts:
                        autotext.set_color('#ffffff')
                        autotext.set_fontsize(12)
                        autotext.set_weight('bold')
                        autotext.set_family('monospace')
                    ax1.set_facecolor('#1a1f2e')
                    fig1.patch.set_facecolor('#0f1419')
                    st.pyplot(fig1)
                
                # 2. Bar Chart - Shape Count
                with chart_col2:
                    st.markdown("""
                    <p style='color: #764ba2; font-weight: 700; font-size: 16px; margin-bottom: 10px;'>📊 Shape Count</p>
                    """, unsafe_allow_html=True)
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    shape_names = list(shape_types.keys())
                    shape_counts = list(shape_types.values())
                    bars = ax2.bar(shape_names, shape_counts, color=['#667eea', '#764ba2', '#00d4ff', '#f093fb', '#f5576c', '#ffa726'][:len(shape_types)], edgecolor='#ffffff', linewidth=2)
                    ax2.set_ylabel('Count', fontsize=12, color='#ffffff', weight='bold', family='monospace')
                    ax2.set_xlabel('Shape Type', fontsize=12, color='#ffffff', weight='bold', family='monospace')
                    ax2.tick_params(colors='#ffffff', labelsize=10)
                    ax2.set_facecolor('#1a1f2e')
                    fig2.patch.set_facecolor('#0f1419')
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', color='#00d4ff', fontweight='bold', fontsize=11, family='monospace')
                    ax2.spines['left'].set_color('#667eea')
                    ax2.spines['bottom'].set_color('#667eea')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    st.pyplot(fig2)
                
                # 3. Area Distribution Chart
                st.markdown("<br>", unsafe_allow_html=True)
                area_col1, area_col2 = st.columns(2)
                
                with area_col1:
                    st.markdown("""
                    <p style='color: #00d4ff; font-weight: 700; font-size: 16px; margin-bottom: 10px;'>📈 Area by Shape</p>
                    """, unsafe_allow_html=True)
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    areas_by_shape = {}
                    for r in results:
                        shape = r['Shape']
                        area = r['Area (pixels)']
                        if shape not in areas_by_shape:
                            areas_by_shape[shape] = []
                        areas_by_shape[shape].append(area)
                    
                    shape_names_area = list(areas_by_shape.keys())
                    avg_areas = [sum(areas_by_shape[s]) / len(areas_by_shape[s]) for s in shape_names_area]
                    
                    bars3 = ax3.barh(shape_names_area, avg_areas, color=['#667eea', '#764ba2', '#00d4ff', '#f093fb', '#f5576c', '#ffa726'][:len(shape_names_area)], edgecolor='#ffffff', linewidth=2)
                    ax3.set_xlabel('Average Area (pixels)', fontsize=12, color='#ffffff', weight='bold', family='monospace')
                    ax3.tick_params(colors='#ffffff', labelsize=10)
                    ax3.set_facecolor('#1a1f2e')
                    fig3.patch.set_facecolor('#0f1419')
                    for i, bar in enumerate(bars3):
                        width = bar.get_width()
                        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                                f'{width:.0f}',
                                ha='left', va='center', color='#00d4ff', fontweight='bold', fontsize=11, family='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1f2e', edgecolor='#667eea'))
                    ax3.spines['bottom'].set_color('#667eea')
                    ax3.spines['left'].set_color('#667eea')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)
                    st.pyplot(fig3)
                
                # 4. Perimeter Distribution
                with area_col2:
                    st.markdown("""
                    <p style='color: #f5576c; font-weight: 700; font-size: 16px; margin-bottom: 10px;'>🔄 Perimeter by Shape</p>
                    """, unsafe_allow_html=True)
                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    perimeters_by_shape = {}
                    for r in results:
                        shape = r['Shape']
                        perim = r['Perimeter (pixels)']
                        if shape not in perimeters_by_shape:
                            perimeters_by_shape[shape] = []
                        perimeters_by_shape[shape].append(perim)
                    
                    shape_names_perim = list(perimeters_by_shape.keys())
                    avg_perims = [sum(perimeters_by_shape[s]) / len(perimeters_by_shape[s]) for s in shape_names_perim]
                    
                    bars4 = ax4.bar(shape_names_perim, avg_perims, color=['#667eea', '#764ba2', '#00d4ff', '#f093fb', '#f5576c', '#ffa726'][:len(shape_names_perim)], edgecolor='#ffffff', linewidth=2)
                    ax4.set_ylabel('Average Perimeter (pixels)', fontsize=12, color='#ffffff', weight='bold', family='monospace')
                    ax4.tick_params(colors='#ffffff', labelsize=10)
                    ax4.set_facecolor('#1a1f2e')
                    fig4.patch.set_facecolor('#0f1419')
                    for bar in bars4:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}',
                                ha='center', va='bottom', color='#f5576c', fontweight='bold', fontsize=11, family='monospace')
                    ax4.spines['left'].set_color('#667eea')
                    ax4.spines['bottom'].set_color('#667eea')
                    ax4.spines['right'].set_visible(False)
                    ax4.spines['top'].set_visible(False)
                    st.pyplot(fig4)
                
                st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                    <p style='margin: 0; color: #a0a0a0; font-size: 13px; line-height: 1.8;'>
                    📊 <span style='color: #667eea; font-weight: 600;'>Chart Insights:</span> These visualizations provide a comprehensive analysis of your shapes. 
                    The pie chart shows proportional distribution, bar charts display counts and averages, 
                    and the horizontal chart reveals area variations across shape types.<br>
                    <span style='color: #00d4ff; font-weight: 600;'>→ Use these to identify patterns and anomalies in your image composition</span>
                    </p>
                    </div>
                """, unsafe_allow_html=True)
        
        # ===== CONTOUR ANALYSIS SECTION =====
        st.markdown("""
        <div class='subheader-custom'>🔲 Contour Analysis</div>
        """, unsafe_allow_html=True)
        
        with st.expander("🎯 Detailed Contour Properties", expanded=False):
            st.markdown("""
            <p style='color: #a0a0a0; font-size: 13px; margin-bottom: 15px;'>
            Contours are the boundaries of shapes detected in the image. Each contour has unique properties that help identify and classify the shape.
            </p>
            """, unsafe_allow_html=True)
            
            # Create contour details table
            contour_details = []
            for i, cnt in enumerate(contours):
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                area = cv2.contourArea(cnt)
                vertices = len(approx)
                
                # Calculate convex hull
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                contour_details.append({
                    "ID": i + 1,
                    "Vertices": vertices,
                    "Contour Length": round(perimeter, 2),
                    "Area": round(area, 2),
                    "Solidity": round(solidity, 4),
                    "Approximation Points": len(cnt)
                })
            
            if contour_details:
                df_contours = pd.DataFrame(contour_details)
                st.dataframe(df_contours.style.highlight_max(axis=0, color='#764ba2').format(precision=3), use_container_width=True)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #667eea; padding: 12px; border-radius: 8px; margin-top: 15px;'>
            <p style='margin: 0; color: #a0a0a0; font-size: 12px; line-height: 1.6;'>
            <span style='color: #667eea; font-weight: 600;'>ℹ️ Contour Properties:</span><br>
            • <strong>Vertices:</strong> Number of corner points in the approximated shape<br>
            • <strong>Contour Length:</strong> Total perimeter of the detected boundary<br>
            • <strong>Solidity:</strong> Ratio of area to convex hull area (1.0 = perfectly convex)<br>
            • <strong>Approximation Points:</strong> Total points defining the contour boundary
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== FEATURE EXTRACTION SECTION =====
        st.markdown("""
        <div class='subheader-custom'>✨ Feature Extraction Analysis</div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Geometric Feature Extraction", expanded=False):
            st.markdown("""
            <p style='color: #a0a0a0; font-size: 13px; margin-bottom: 15px;'>
            Feature extraction identifies and quantifies geometric properties of detected shapes. These metrics are crucial for shape classification and analysis.
            </p>
            """, unsafe_allow_html=True)
            
            # Calculate features for each contour
            features_list = []
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # Circularity (how round the shape is)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                else:
                    circularity = 0
                
                # Aspect ratio and compactness
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Compactness (perimeter normalized by area)
                compactness = perimeter / np.sqrt(area) if area > 0 else 0
                
                # Convex hull
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Extent (area to bounding rectangle ratio)
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0
                
                features_list.append({
                    "Shape ID": i + 1,
                    "Circularity": round(circularity, 4),
                    "Aspect Ratio": round(aspect_ratio, 4),
                    "Compactness": round(compactness, 4),
                    "Solidity": round(solidity, 4),
                    "Extent": round(extent, 4)
                })
            
            if features_list:
                df_features = pd.DataFrame(features_list)
                st.dataframe(df_features.style.highlight_max(axis=0, color='#00d4ff').format(precision=4), use_container_width=True)
                
                # Feature distribution charts
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <p style='color: #667eea; font-weight: 700; font-size: 16px; margin-bottom: 10px;'>📊 Feature Distributions</p>
                """, unsafe_allow_html=True)
                
                feat_col1, feat_col2 = st.columns(2)
                
                # Circularity distribution
                with feat_col1:
                    st.markdown("""
                    <p style='color: #764ba2; font-weight: 700; font-size: 14px; margin-bottom: 8px;'>🎯 Circularity Distribution</p>
                    """, unsafe_allow_html=True)
                    fig_circ, ax_circ = plt.subplots(figsize=(8, 5))
                    circularities = [f['Circularity'] for f in features_list]
                    shape_ids = [f['Shape ID'] for f in features_list]
                    bars_circ = ax_circ.bar(shape_ids, circularities, color=['#667eea' if c > 0.85 else '#f5576c' for c in circularities], edgecolor='#ffffff', linewidth=2)
                    ax_circ.axhline(y=0.85, color='#00d4ff', linestyle='--', linewidth=2, label='Circle Threshold (0.85)')
                    ax_circ.set_ylabel('Circularity', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_circ.set_xlabel('Shape ID', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_circ.set_ylim([0, 1.1])
                    ax_circ.tick_params(colors='#ffffff', labelsize=10)
                    ax_circ.legend(fontsize=9, loc='upper right', framealpha=0.9)
                    ax_circ.set_facecolor('#1a1f2e')
                    fig_circ.patch.set_facecolor('#0f1419')
                    ax_circ.spines['left'].set_color('#667eea')
                    ax_circ.spines['bottom'].set_color('#667eea')
                    ax_circ.spines['right'].set_visible(False)
                    ax_circ.spines['top'].set_visible(False)
                    for bar, val in zip(bars_circ, circularities):
                        ax_circ.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                                    f'{val:.3f}', ha='center', va='bottom', color='#00d4ff', fontweight='bold', fontsize=10, family='monospace')
                    st.pyplot(fig_circ)
                
                # Aspect Ratio distribution
                with feat_col2:
                    st.markdown("""
                    <p style='color: #00d4ff; font-weight: 700; font-size: 14px; margin-bottom: 8px;'>📐 Aspect Ratio Distribution</p>
                    """, unsafe_allow_html=True)
                    fig_aspect, ax_aspect = plt.subplots(figsize=(8, 5))
                    aspect_ratios = [f['Aspect Ratio'] for f in features_list]
                    bars_aspect = ax_aspect.bar(shape_ids, aspect_ratios, color=['#667eea' if 0.9 <= ar <= 1.1 else '#f093fb' for ar in aspect_ratios], edgecolor='#ffffff', linewidth=2)
                    ax_aspect.axhline(y=1.0, color='#00d4ff', linestyle='--', linewidth=2, label='Square (1.0)')
                    ax_aspect.set_ylabel('Aspect Ratio', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_aspect.set_xlabel('Shape ID', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_aspect.tick_params(colors='#ffffff', labelsize=10)
                    ax_aspect.legend(fontsize=9, loc='upper right', framealpha=0.9)
                    ax_aspect.set_facecolor('#1a1f2e')
                    fig_aspect.patch.set_facecolor('#0f1419')
                    ax_aspect.spines['left'].set_color('#667eea')
                    ax_aspect.spines['bottom'].set_color('#667eea')
                    ax_aspect.spines['right'].set_visible(False)
                    ax_aspect.spines['top'].set_visible(False)
                    for bar, val in zip(bars_aspect, aspect_ratios):
                        ax_aspect.text(bar.get_x() + bar.get_width()/2., val + 0.05,
                                      f'{val:.2f}', ha='center', va='bottom', color='#00d4ff', fontweight='bold', fontsize=10, family='monospace')
                    st.pyplot(fig_aspect)
                
                # Solidity and Extent comparison
                st.markdown("<br>", unsafe_allow_html=True)
                sol_col1, sol_col2 = st.columns(2)
                
                with sol_col1:
                    st.markdown("""
                    <p style='color: #f5576c; font-weight: 700; font-size: 14px; margin-bottom: 8px;'>🔒 Solidity Index</p>
                    """, unsafe_allow_html=True)
                    fig_solid, ax_solid = plt.subplots(figsize=(8, 5))
                    solidities = [f['Solidity'] for f in features_list]
                    bars_solid = ax_solid.barh(shape_ids, solidities, color=['#667eea' if s > 0.8 else '#ffa726' for s in solidities], edgecolor='#ffffff', linewidth=2)
                    ax_solid.set_xlabel('Solidity', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_solid.set_xlim([0, 1.1])
                    ax_solid.tick_params(colors='#ffffff', labelsize=10)
                    ax_solid.set_facecolor('#1a1f2e')
                    fig_solid.patch.set_facecolor('#0f1419')
                    ax_solid.spines['bottom'].set_color('#667eea')
                    ax_solid.spines['left'].set_color('#667eea')
                    ax_solid.spines['right'].set_visible(False)
                    ax_solid.spines['top'].set_visible(False)
                    for i, bar in enumerate(bars_solid):
                        width = bar.get_width()
                        ax_solid.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                                    f'{solidities[i]:.3f}', ha='left', va='center', color='#00d4ff', fontweight='bold', fontsize=10, family='monospace')
                    st.pyplot(fig_solid)
                
                with sol_col2:
                    st.markdown("""
                    <p style='color: #ffa726; font-weight: 700; font-size: 14px; margin-bottom: 8px;'>📦 Extent Ratio</p>
                    """, unsafe_allow_html=True)
                    fig_extent, ax_extent = plt.subplots(figsize=(8, 5))
                    extents = [f['Extent'] for f in features_list]
                    bars_extent = ax_extent.barh(shape_ids, extents, color=['#667eea' if e > 0.7 else '#f5576c' for e in extents], edgecolor='#ffffff', linewidth=2)
                    ax_extent.set_xlabel('Extent', fontsize=11, color='#ffffff', weight='bold', family='monospace')
                    ax_extent.set_xlim([0, 1.1])
                    ax_extent.tick_params(colors='#ffffff', labelsize=10)
                    ax_extent.set_facecolor('#1a1f2e')
                    fig_extent.patch.set_facecolor('#0f1419')
                    ax_extent.spines['bottom'].set_color('#667eea')
                    ax_extent.spines['left'].set_color('#667eea')
                    ax_extent.spines['right'].set_visible(False)
                    ax_extent.spines['top'].set_visible(False)
                    for i, bar in enumerate(bars_extent):
                        width = bar.get_width()
                        ax_extent.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                                    f'{extents[i]:.3f}', ha='left', va='center', color='#00d4ff', fontweight='bold', fontsize=10, family='monospace')
                    st.pyplot(fig_extent)
                
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #00d4ff; padding: 12px; border-radius: 8px; margin-top: 15px;'>
                <p style='margin: 0; color: #a0a0a0; font-size: 12px; line-height: 1.6;'>
                <span style='color: #00d4ff; font-weight: 600;'>🎯 Feature Definitions:</span><br>
                • <strong>Circularity:</strong> Measures roundness (1.0 = perfect circle, lower = irregular)<br>
                • <strong>Aspect Ratio:</strong> Width/Height ratio (1.0 = square, >1 = wider, <1 = taller)<br>
                • <strong>Compactness:</strong> Perimeter normalized by area (lower = more compact)<br>
                • <strong>Solidity:</strong> Area/ConvexHull ratio (1.0 = no concavity)<br>
                • <strong>Extent:</strong> Area/BoundingBox ratio (higher = fills bounding box better)
                </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer-custom'>
        <p style='margin: 0; font-size: 14px; color: #a0a0a0;'>
            <span style='color: #667eea; font-weight: 600;'>⚡ Powered by</span> OpenCV & Streamlit 
            <span style='color: #667eea; font-weight: 600;'>|</span> Computer Vision Assignment
        </p>
        <p style='margin: 8px 0 0 0; font-size: 12px; color: #666;'>
            🎨 Enhanced with Modern UI & Animations
        </p>
    </div>
    """, unsafe_allow_html=True)