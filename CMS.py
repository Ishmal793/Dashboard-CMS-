import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

# Function to load default data
@st.cache_data
def load_default_data():
    return pd.read_excel(
        'Modified_PPE_compliance_dataset.xlsx',
        sheet_name='Sheet1',
        engine='openpyxl'
    )

# Function to load uploaded files (supports Excel and CSV)
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

# Sidebar for file upload or default dataset
st.sidebar.title("Upload or Load Dataset")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

# Load dataset based on user input
if data_source == "Default Dataset":
    data = load_default_data()
    st.sidebar.success("Default dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a dataset to proceed.")
        st.stop()

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date']).dt.date


# Define color palettes
default_colors = px.colors.qualitative.Plotly
time_series_colors = px.colors.qualitative.Set2

selected_analysis = st.sidebar.radio(
    "Select Analysis Level:",
    [
        "Variable Analytics",
        "Strategic Insights"
    ]
)


# Sidebar Filters

# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_set_query_params()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)
analysis_variable=""
selected_insight=""
# Logic for Variable Analytics
if selected_analysis == "Variable Analytics":
    # Radio Button for Variable-Based Analytics
    analysis_variable = st.sidebar.radio(
        "Select Variable for Analysis:",
        [
            "Over All",
            "Analytics of Employee and Employees",
            "Analytics of Unit and Units",
            "Analytics of Shift and Shifts",
            "Analytics of Time Series",
            "Analytics of Camera Units"
        ]
    )



# Logic for Strategic Insights
elif selected_analysis == "Strategic Insights":
    # Radio Buttons for Selecting Insights
    selected_insight = st.sidebar.radio(
        "Select an Insight:",
        [
            "Combined Insights",
            "Critical Zone Insights",
            "Targets Monitoring Insights",
            "Time Tracking Insights",
            "Shift Productivity Insights",
            "Predictive Insights",
            "Growth Tracker Insights",
            "Risk Radar Insights",
            "Association Insights",

        ]
    )
# Data Filtering
st.sidebar.header("Filters")
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Compliance", "Violation"])

min_date, max_date = min(data['Date']), max(data['Date'])
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start Date cannot be after End Date")
# Sidebar Filters
employee = st.sidebar.multiselect('Select Employee', options=data['Employee_Name'].unique())
shift = st.sidebar.multiselect('Select Shift', options=data['Shift'].unique())
factory = st.sidebar.multiselect('Select Factory', options=data['Factory'].unique())
department = st.sidebar.multiselect('Select Department', options=data['Department'].unique())
camera = st.sidebar.multiselect('Select Camera', options=data['Camera'].unique())
violation_type = st.sidebar.multiselect('Select Violation Type', options=data['Violation_Type'].unique())  # New filter for violation types

# Render charts based on the selected variable
filtered_data = data[
    (data['Date'] >= start_date) & (data['Date'] <= end_date) &
    (data['Employee_Name'].isin(employee) if employee else True) &
    (data['Shift'].isin(shift) if shift else True) &
    (data['Factory'].isin(factory) if factory else True) &
    (data['Department'].isin(department) if department else True) &
    (data['Camera'].isin(camera) if camera else True) &
    (data['Violation_Type'].isin(violation_type) if violation_type else True)  # Apply filter for violation types
]
# Determine the relevant data based on the analysis type
if analysis_type == "Violation":
    relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
    current_rate = (relevant_data.shape[0] / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Violation Rate"
    relevant_checks = relevant_data.shape[0]
else:
    relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
    compliant_checks = relevant_data.shape[0]
    current_rate = (compliant_checks / filtered_data.shape[0] * 100) if filtered_data.shape[0] > 0 else 0
    rate_label = "Current Compliance Rate"
    relevant_checks = compliant_checks


if relevant_data.shape[0] > 0:
    # Extract month and group data to calculate monthly rates
    relevant_data['Month'] = pd.to_datetime(relevant_data['Date']).dt.to_period('M').astype(str)
    monthly_rate = relevant_data.groupby('Month')['Violation_Type'].apply(
        lambda x: (x == 'Compliant').sum() / len(x) * 100
        if analysis_type == "Compliant" else (x != 'Compliant').sum() / len(x) * 100
    ).reset_index(name='Rate')

    print("Monthly Rates:")
    print(monthly_rate)

    # Check if all rates are identical (e.g., all 100%)
    if monthly_rate['Rate'].nunique() == 1:
        next_month_prediction = current_rate  # Maintain current rate if historical data is uniform
    elif len(monthly_rate) > 2:
        # Use Holt-Winters with a damped trend to avoid explosive predictions
        model = ExponentialSmoothing(
            monthly_rate['Rate'],
            trend='add',
            damped_trend=True,  # Dampen the trend
            seasonal=None,
            initialization_method='estimated'
        ).fit()

        # Predict the next month's rate
        next_month_prediction = model.forecast(1).values[0]
    elif len(monthly_rate) > 1:
        # Fall back to linear regression if only two months are available
        coeffs = np.polyfit(range(len(monthly_rate)), monthly_rate['Rate'], 1)
        next_month_prediction = coeffs[0] * len(monthly_rate) + coeffs[1]
    else:
        # Default to current rate if only one month of data is available
        next_month_prediction = monthly_rate['Rate'].iloc[-1] if len(monthly_rate) == 1 else current_rate

    # Ensure the prediction is between 0% and 100%
    next_month_prediction = min(max(next_month_prediction, 0), 100)

else:
    # Default prediction if no data available
    next_month_prediction = current_rate

# Output the results
print(f"{rate_label}: {current_rate:.2f}%")
print(f"Next month's prediction: {next_month_prediction:.2f}%")

 # Function to create pie charts
def create_pie_chart(data, group_by, title):
    pie_data = data[group_by].value_counts().reset_index()
    pie_data.columns = [group_by, 'Count']
    fig = px.pie(pie_data, names=group_by, values='Count', title=title)
    st.plotly_chart(fig, use_container_width=True)

# Function to create bar charts
def create_bar_chart(data, group_by, title, color_palette):
    grouped_data = data.groupby(group_by).size().reset_index(name='Count')
    grouped_data['Color'] = grouped_data.index.map(lambda x: color_palette[x % len(color_palette)])

    fig = px.bar(
        grouped_data, x=group_by, y='Count',
        title=title, color='Color',
        labels={'Count': 'Total Count'},
        color_discrete_sequence=color_palette
    )
    st.plotly_chart(fig, use_container_width=True)



    # Display the breakdown chart
    st.plotly_chart(fig_camera_breakdown, use_container_width=True)
if selected_analysis == "Variable Analytics":
    if analysis_variable == "Over All":
        # Select relevant data based on the analysis type
        total_checks = filtered_data.shape[0]

        # Display Header and Metrics
        st.header(f"Overall {analysis_type} Dashboard")

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        # Display Current Rate, Next Month Prediction, Total Checks, and Relevant Checks
        col1.metric(rate_label, f"{current_rate:.2f}%")
        col2.metric("Next Month Prediction", f"{next_month_prediction:.2f}%")
        col3.metric("Total Checks", total_checks)
        col4.metric("Relevant Checks", relevant_checks)
        # Group data for visualizations
        if analysis_type == "Violation":
            grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
                Total_Violations=('Violation_Type', 'count')
            ).reset_index()
        else:
            grouped_data = relevant_data.groupby(['Factory', 'Shift']).agg(
                Total_Compliance=('Violation_Type', 'count')
            ).reset_index()

        # Factory-wise Violations/Compliance Gauge
        st.subheader(f"{analysis_type} by Factory")

        factory_colors = ['#00FF00', '#FF4500', '#1E90FF', '#FFFF00',
                          '#FF1493']  # Green, OrangeRed, DodgerBlue, Yellow, DeepPink (avoiding pink now)

        # Factory-wise Violations/Compliance Gauge
        col1, col2, col3 = st.columns(3)
        for i, (factory, count) in enumerate(grouped_data.groupby('Factory')[
                                                 f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum().items(),
                                             1):
            with [col1, col2, col3][i % 3]:
                color_index = i % len(factory_colors)  # Cycle through the color palette
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=count,
                    title={"text": f"Factory {factory} {analysis_type}"},
                    gauge={
                        'axis': {'range': [0, max(grouped_data.groupby('Factory')[
                                                      f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                        'bar': {'color': factory_colors[color_index]}
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

        # Shift-wise Violations/Compliance Gauge
        st.subheader(f"{analysis_type} by Shift")

        col4, col5 = st.columns(2)
        with col4:
            shift_value = grouped_data[grouped_data['Shift'] == 'Morning'][
                f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
            fig_morning = go.Figure(go.Indicator(
                mode="gauge+number",
                value=shift_value,
                title={"text": "Morning Shift"},
                gauge={
                    'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                                  f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                    'bar': {'color': '#32CD32'}  # LimeGreen color for Morning Shift
                }
            ))

            st.plotly_chart(fig_morning, use_container_width=True)

        with col5:
            shift_value = grouped_data[grouped_data['Shift'] == 'Evening'][
                f'Total_Violations' if analysis_type == "Violation" else 'Total_Compliance'].sum()
            fig_evening = go.Figure(go.Indicator(
                mode="gauge+number",
                value=shift_value,
                title={"text": "Evening Shift"},
                gauge={
                    'axis': {'range': [0, max(grouped_data.groupby('Shift')[
                                                  f'Total_Violations' if analysis_type == 'Violation' else 'Total_Compliance'].sum())]},
                    'bar': {'color': '#FF8C00'}  # DarkOrange for Evening Shift
                }
            ))

            st.plotly_chart(fig_evening, use_container_width=True)
        row_selection = st.radio("Choose Rows to Display:", ("First Five Rows", "Last Five Rows"))

        # Display data based on radio selection
        if row_selection == "First Five Rows":
            st.write("### First Five Rows of the Dataset")
            st.write(data.head())
        else:
            st.write("### Last Five Rows of the Dataset")
            st.write(data.tail())


    elif analysis_variable == "Analytics of Employee and Employees":
        st.header("Employee Analytics")

        if analysis_type == "Violation":
            relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
            title = "Employee Violations"
        else:
            relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']
            title = "Employee Compliance"

        # Group data by Employee to count occurrences (first chart)
        employee_counts = relevant_data['Employee_Name'].value_counts().reset_index()
        employee_counts.columns = ['Employee_Name', 'Count']

        # Create a bar chart: Total Compliance/Violations by Employee
        fig_employee_compliance = px.bar(
            employee_counts,
            x='Employee_Name',
            y='Count',
            title=title,
            labels={'Count': 'Total Count'},
            color='Employee_Name',  # Assign distinct colors to each employee
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use a qualitative palette
        )

        # Update layout of the first chart for better appearance
        fig_employee_compliance.update_layout(
            xaxis_title="Employee Name",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Second Chart: Breakdown by Violation/Compliance Type for Each Employee
        breakdown_data = relevant_data.groupby(['Employee_Name', 'Violation_Type']).size().reset_index(name='Count')

        fig_violation_breakdown = px.bar(
            breakdown_data,
            x='Employee_Name',
            y='Count',
            color='Violation_Type',  # Distinguish by Violation/Compliance Type
            title=f"{title} by Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use a distinct color palette for types
        )

        # Update layout of the second chart for better appearance
        fig_violation_breakdown.update_layout(
            xaxis_title="Employee Name",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for good contrast
        )

        # Display both charts
        st.plotly_chart(fig_employee_compliance, use_container_width=True)
        st.plotly_chart(fig_violation_breakdown, use_container_width=True)


    elif analysis_variable == "Analytics of Unit and Units":
        st.header("Unit Analytics")
        st.header(f"{analysis_type} by Unit")

        # Define distinct colors for factories
        factory_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'  # Cyan
        ]

        # Factory-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Factory")

        factory_data = relevant_data.groupby(['Factory']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Create a color map for factories
        factory_data['Color'] = factory_data.index.map(lambda x: factory_colors[x % len(factory_colors)])

        fig_factory = px.bar(factory_data, x='Factory', y='Total_Count',
                             title=f"{analysis_type} by Factory",
                             labels={
                                 'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                             color='Color')  # Use the assigned color

        # Update layout for integer x-axis ticks
        fig_factory.update_layout(
            xaxis=dict(
                dtick=1,  # Set the tick interval to 1 for integer values
                tickmode='linear'  # Ensure ticks are linear
            )
        )

        st.plotly_chart(fig_factory, use_container_width=True)

        # Department-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Department")

        department_data = relevant_data.groupby(['Department']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Define distinct colors for departments
        department_colors = [
            '#ffbb78',  # Light Orange
            '#98df8a',  # Light Green
            '#ff9896',  # Light Red
            '#c5b0d5',  # Light Purple
            '#f7b6d2',  # Light Pink
            '#c49c94',  # Light Brown
            '#f7f7f7',  # Light Gray
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
            '#f3d9a4'  # Light Yellow
        ]

        # Create a color map for departments
        department_data['Color'] = department_data.index.map(lambda x: department_colors[x % len(department_colors)])

        fig_department = px.bar(department_data, x='Department', y='Total_Count',
                                title=f"{analysis_type} by Department",
                                labels={
                                    'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                                color='Color')  # Use the assigned color

        # Update layout for integer x-axis ticks
        fig_department.update_layout(
            xaxis=dict(
                dtick=1,  # Set the tick interval to 1 for integer values
                tickmode='linear'  # Ensure ticks are linear
            )
        )

        st.plotly_chart(fig_department, use_container_width=True)
        # Visualization Logic for Violations
        if analysis_type == "Violation":
            # Filter for Violations
            violation_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']

            # Create a Bar Chart for Violations
            if not violation_data.empty:
                violation_count = violation_data['Violation_Type'].value_counts().reset_index()
                violation_count.columns = ['Violation Type', 'Count']

                # Define distinct colors for violation types using qualitative colors
                fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                       title="Violation Counts",
                                       labels={'Count': 'Number of Violations'},
                                       color='Violation Type',  # Color by Violation Type for distinct colors
                                       color_discrete_sequence=px.colors.qualitative.Dark2)  # Darker color palette

                st.plotly_chart(fig_violation, use_container_width=True)
            else:
                st.write("No violation data available for the selected filters.")

    elif analysis_variable == "Analytics of Shift and Shifts":
        st.header("Shift Analytics")
        # Shift-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Shift")

        # Group data by shift to calculate total counts
        shift_data = relevant_data.groupby(['Shift']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Bar chart: Overall Violations/Compliance by Shift
        fig_shift = px.bar(
            shift_data,
            x='Shift',
            y='Total_Count',
            title=f"{analysis_type} by Shift",
            labels={
                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'
            },
            color='Shift',  # Assign distinct colors for each shift
            color_discrete_sequence=px.colors.qualitative.Plotly  # Use a vibrant color palette
        )

        # Update layout for better appearance
        fig_shift.update_layout(
            xaxis_title="Shift",
            yaxis_title="Total Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Display the shift-wise chart
        st.plotly_chart(fig_shift, use_container_width=True)

        # Shift-wise Breakdown by Type (Second Chart)
        st.subheader(f"{analysis_type} Breakdown by Shift and Type")

        # Group data by Shift and Violation/Compliance Type
        shift_breakdown = relevant_data.groupby(['Shift', 'Violation_Type']).size().reset_index(name='Count')

        # Bar chart: Breakdown of Types by Shift
        fig_shift_breakdown = px.bar(
            shift_breakdown,
            x='Shift',
            y='Count',
            color='Violation_Type',  # Color by type for distinction
            title=f"{analysis_type} Breakdown by Shift and Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative palette for better distinction
        )

        # Update layout for better appearance
        fig_shift_breakdown.update_layout(
            xaxis_title="Shift",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for good contrast
        )

        # Display the shift-wise breakdown chart
        st.plotly_chart(fig_shift_breakdown, use_container_width=True)


    elif analysis_variable == "Analytics of Time Series" :

        # Date-wise Violations/Compliance Chart (Line Chart)
        st.subheader(f"{analysis_type} by Date")

        # Group data by Date to calculate total counts
        date_data = relevant_data.groupby(['Date']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Line chart: Overall Violations/Compliance Over Time
        fig_date = px.line(
            date_data,
            x='Date',
            y='Total_Count',
            title=f"{analysis_type} Over Time",
            labels={
                'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'
            },
            markers=True,  # Add markers for better visibility
            color_discrete_sequence=px.colors.qualitative.Set1  # Use Set1 color scheme
        )

        # Update layout for better appearance
        fig_date.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white"),  # White font for contrast
        )

        # Display the line chart
        st.plotly_chart(fig_date, use_container_width=True)

        # Breakdown by Type Over Time (Area Chart)
        st.subheader(f"{analysis_type} Breakdown by Date and Type")

        # Group data by Date and Violation/Compliance Type
        date_breakdown = relevant_data.groupby(['Date', 'Violation_Type']).size().reset_index(name='Count')

        # Area chart: Breakdown by Violation/Compliance Type Over Time
        fig_date_breakdown = px.area(
            date_breakdown,
            x='Date',
            y='Count',
            color='Violation_Type',  # Distinguish by type
            title=f"{analysis_type} Breakdown Over Time",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative palette
        )

        # Update layout for better appearance
        fig_date_breakdown.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font for contrast
        )

        # Display the area chart
        st.plotly_chart(fig_date_breakdown, use_container_width=True)



    elif analysis_variable == "Analytics of Camera Units":
        st.header("Camera Unit Analytics")

        # Camera-wise Violations/Compliance Chart
        st.subheader(f"{analysis_type} by Camera")

        # Grouping data by Camera to count occurrences of violations
        camera_data = relevant_data.groupby(['Camera']).agg(
            Total_Violations=(
            'Violation_Type', lambda x: (x != 'Compliant').sum()) if analysis_type == "Violation" else (
                'Employee_Name', 'count')
        ).reset_index()

        # Create a bar chart for total violations by camera
        fig_camera = px.bar(
            camera_data,
            x='Camera',
            y='Total_Violations',
            title=f"Total {analysis_type} by Camera",
            labels={'Total_Violations': 'Number of Violations'},
            color='Total_Violations',  # Color by Total_Violations for distinct colors
            color_continuous_scale='YlOrRd'  # Yellow-Orange-Red color scale
        )

        # Update layout for better appearance
        fig_camera.update_layout(
            xaxis_title="Camera",
            yaxis_title="Number of Violations",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
            font=dict(color="black")  # Black font color for good contrast
        )

        # Display the chart
        st.plotly_chart(fig_camera, use_container_width=True)

        # Breakdown by Type Over Camera (Second Chart)
        st.subheader(f"{analysis_type} Breakdown by Camera and Type")

        # Grouping data by Camera and Violation Type to count occurrences
        camera_breakdown = relevant_data.groupby(['Camera', 'Violation_Type']).size().reset_index(name='Count')

        # Create a grouped bar chart for breakdown by type for each camera
        fig_camera_breakdown = px.bar(
            camera_breakdown,
            x='Camera',
            y='Count',
            color='Violation_Type',  # Distinguish by type
            title=f"{analysis_type} Breakdown by Camera and Type",
            labels={'Count': 'Total Count', 'Violation_Type': 'Type'},
            barmode='group',  # Group bars for better comparison
            color_discrete_sequence=px.colors.qualitative.Set2  # Use another qualitative color palette
        )

        # Update layout for better appearance
        fig_camera_breakdown.update_layout(
            xaxis_title="Camera",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for better appearance
            font=dict(color="black")  # Black font for good contrast
        )
        st.plotly_chart(fig_camera_breakdown, use_container_width=True)
elif selected_analysis == "Strategic Insights":
    # Combined Insights
    if selected_insight == "Combined Insights":
        st.subheader(f"{analysis_type} by Factory, Department")

        # Group data for Unit-wise Violations/Compliance
        grouped_unit_data = relevant_data.groupby(['Factory', 'Department']).agg(
            Total_Count=('Violation_Type', 'count') if analysis_type == "Violation" else ('Employee_Name', 'count')
        ).reset_index()

        # Color palette for the bar chart
        color_palette = px.colors.qualitative.Set3  # Using a vibrant color palette

        # Create a bar chart for Unit-wise Violations/Compliance
        fig_unit = px.bar(grouped_unit_data,
                          x='Factory',
                          y='Total_Count',
                          color='Department',
                          title=f"{analysis_type} by Unit",
                          labels={
                              'Total_Count': 'Total Violations' if analysis_type == "Violation" else 'Total Compliance'},
                          color_discrete_sequence=color_palette)  # Assign color palette

        # Update layout for better appearance
        fig_unit.update_layout(
            xaxis_title="Factory",
            yaxis_title="Count",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color="white")  # White font color for contrast with dark background
        )

        st.plotly_chart(fig_unit, use_container_width=True)

        # Filter data based on user input
        if analysis_type == "Violation":
            relevant_data = data[data['Violation_Type'] != 'Compliant']
        else:
            relevant_data = data[data['Violation_Type'] == 'Compliant']


        # Function to create combined charts
        def combined_charts():
            fig = go.Figure()

            # Group data by Department and Shift
            department_shift_data = relevant_data.groupby(['Department', 'Shift']).agg(
                Total_Count=('Violation_Type', 'count')
            ).reset_index()

            # Add Department by Shift Bar Chart
            # Using distinct colors for each department
            department_colors = px.colors.qualitative.Pastel  # Color palette for department-wise chart

            for i, department in enumerate(department_shift_data['Department'].unique()):
                department_data = department_shift_data[department_shift_data['Department'] == department]
                color = department_colors[i % len(department_colors)]  # Cycle through the color palette
                fig.add_trace(go.Bar(
                    x=department_data['Shift'],
                    y=department_data['Total_Count'],
                    name=str(department),  # Ensure department name is used as legend entry
                    hoverinfo='text',
                    text=department_data['Total_Count'],
                    marker_color=color  # Set the color for each department
                ))

            # Update the layout with title and axis labels
            fig.update_layout(
                title=f"{analysis_type} by Department and Shift",
                barmode='stack',  # Stacked bar chart
                xaxis_title='Shift',
                yaxis_title='Total Count',
                legend_title='Department',
                template='plotly_white',  # White background for better visibility
                font=dict(color="white")  # White font for contrast
            )

            # Render the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)


        # Call the function to display combined charts
        combined_charts()
        # Factory by Trend Over Time
        trend_data = relevant_data.groupby(['Date', 'Factory']).agg(
            Total_Count=('Violation_Type', 'count')
        ).reset_index()

        # Create a new figure for the trend chart
        fig_trend = go.Figure()

        # Use a color palette for the Factory trends
        factory_colors = px.colors.qualitative.Vivid  # Vivid colors for factory trends

        for i, factory in enumerate(trend_data['Factory'].unique()):
            factory_data = trend_data[trend_data['Factory'] == factory]
            color = factory_colors[i % len(factory_colors)]  # Assign color to each factory
            fig_trend.add_trace(go.Scatter(
                x=factory_data['Date'],
                y=factory_data['Total_Count'],
                mode='lines+markers',
                name=str(factory),  # Ensure name is a string
                hoverinfo='text',
                text=factory_data['Total_Count'],
                line=dict(color=color, width=2),  # Set line color
                marker=dict(size=6, symbol='circle', color=color)  # Set marker color
            ))

        fig_trend.update_layout(
            title=f"{analysis_type} Trend Over Time by Factory",
            xaxis_title='Date',
            yaxis_title='Total Count',
            legend_title='Factory',
            template='plotly_white',  # Keep background white
            plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # Employee by Over Time
        employee_data = relevant_data.groupby(['Date', 'Employee_Name']).agg(
            Total_Count=('Violation_Type', 'count')
        ).reset_index()

        # Create a new figure for the employee trend chart
        fig_employee = go.Figure()

        # Use a color palette for Employee trends
        employee_colors = px.colors.qualitative.Set2  # Soft colors for employee trends

        for i, employee in enumerate(employee_data['Employee_Name'].unique()):
            emp_data = employee_data[employee_data['Employee_Name'] == employee]
            color = employee_colors[i % len(employee_colors)]  # Assign color to each employee
            fig_employee.add_trace(go.Scatter(
                x=emp_data['Date'],
                y=emp_data['Total_Count'],
                mode='lines+markers',
                name=str(employee),  # Ensure name is a string
                hoverinfo='text',
                text=emp_data['Total_Count'],
                line=dict(color=color, width=2),  # Set line color
                marker=dict(size=6, symbol='circle', color=color)  # Set marker color
            ))

        fig_employee.update_layout(
            title=f"{analysis_type} Over Time by Employee",
            xaxis_title='Date',
            yaxis_title='Total Count',
            legend_title='Employee',
            template='plotly_white',  # Keep background white
            plot_bgcolor='rgba(0, 0, 0, 0)'  # Ensure the plot area background is transparent
        )

        st.plotly_chart(fig_employee, use_container_width=True)

    # Critical Zone Insights
    elif selected_insight == "Critical Zone Insights":
        zone_colors = {'Green Zone': 'green', 'Yellow Zone': 'yellow', 'Red Zone': 'red'}

        # Filter data based on user input
        if analysis_type == "Violation":
            st.subheader("Critical Zone Insights (Red: > 50% Violation Rate, Yellow: 30-50%, Green: < 30%)")

            # Calculate violation rates for critical zone monitoring
            violation_rates = data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

            # Categorize into critical zones
            violation_rates['Zone'] = pd.cut(violation_rates['Violation Rate'], bins=[0, 30, 50, 100],
                                             labels=['Green Zone', 'Yellow Zone', 'Red Zone'])

            # Display the categorized zones as a dataframe
            st.dataframe(violation_rates)

            # Plot Critical Zone Alerts as a Bar Chart
            fig_critical_zones = px.bar(violation_rates,
                                        x='Department',
                                        y='Violation Rate',
                                        color='Zone',
                                        title="Critical Zone Violation Rates",
                                        labels={'Violation Rate': 'Violation Rate (%)'},
                                        color_discrete_map=zone_colors)  # Using zone color map

            # Update layout for violation rates
            fig_critical_zones.update_layout(
                xaxis_title="Department",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            )

            # Display the violation chart
            st.plotly_chart(fig_critical_zones, use_container_width=True)

        else:
            st.subheader("Critical Zone Insights (Red: < 50% Compliance, Yellow: 50-80%, Green: > 80%)")

            # Calculate compliance rates for critical zone monitoring
            compliance_rates = data.groupby(['Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

            # Categorize into critical zones
            compliance_rates['Zone'] = pd.cut(compliance_rates['Compliance Rate'], bins=[0, 50, 80, 100],
                                              labels=['Red Zone', 'Yellow Zone', 'Green Zone'])

            # Display the categorized zones as a dataframe
            st.dataframe(compliance_rates)

            # Plot Critical Zone Alerts as a Bar Chart
            fig_compliance_zones = px.bar(compliance_rates,
                                          x='Department',
                                          y='Compliance Rate',
                                          color='Zone',
                                          title="Critical Zone Compliance Rates",
                                          labels={'Compliance Rate': 'Compliance Rate (%)'},
                                          color_discrete_map=zone_colors)  # Using zone color map

            # Update layout for compliance rates
            fig_compliance_zones.update_layout(
                xaxis_title="Department",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
                xaxis=dict(tickmode='linear', tick0=1, dtick=1)  # Force x-axis to show integers only
            )

            # Display the compliance chart
            st.plotly_chart(fig_compliance_zones, use_container_width=True)


    # Targets Monitoring Insights
    elif selected_insight == "Targets Monitoring Insights":
        # For Violations

        if analysis_type == "Violation":
            st.subheader("Targets Monitoring - Violation Rate vs Targets")

            violation_data = filtered_data.groupby(['Factory', 'Department']).agg(
                violation_count=('Violation_Type', lambda x: (x != 'Compliant').sum()),
                total=('Violation_Type', 'count')
            ).reset_index()

            violation_data['Target Violation Rate'] = 30  # Assuming target of 30%
            violation_data['Actual Violation Rate'] = (violation_data['violation_count'] / violation_data[
                'total']) * 100

            st.dataframe(violation_data)

            fig_target_violation = px.bar(violation_data, x='Department', y='Actual Violation Rate', color='Factory',
                                          title="Actual vs Target Violation Rates",
                                          labels={'Actual Violation Rate': 'Actual Violation Rate (%)'},
                                          color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig_target_violation.add_scatter(x=violation_data['Department'], y=violation_data['Target Violation Rate'],
                                             mode='lines', name='Target Violation Rate', line=dict(dash='dash'))
            st.plotly_chart(fig_target_violation, use_container_width=True)

        # For Compliance
        else:
            st.subheader("Targets Monitoring - Compliance Rate vs Targets")

            compliance_data = filtered_data.groupby(['Factory', 'Department']).agg(
                compliance_count=('Violation_Type', lambda x: (x == 'Compliant').sum()),
                total=('Violation_Type', 'count')
            ).reset_index()

            compliance_data['Target Compliance Rate'] = 70  # Assuming target of 70%
            compliance_data['Actual Compliance Rate'] = (compliance_data['compliance_count'] / compliance_data[
                'total']) * 100

            st.dataframe(compliance_data)

            fig_target_compliance = px.bar(compliance_data, x='Department', y='Actual Compliance Rate', color='Factory',
                                           title="Actual vs Target Compliance Rates",
                                           labels={'Actual Compliance Rate': 'Actual Compliance Rate (%)'},
                                           color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig_target_compliance.add_scatter(x=compliance_data['Department'],
                                              y=compliance_data['Target Compliance Rate'],
                                              mode='lines', name='Target Compliance Rate', line=dict(dash='dash'))
            st.plotly_chart(fig_target_compliance, use_container_width=True)
        

        # Define target rates
        target_violation_rate = 10  # Target for violation (less than 10% violation rate)
        target_compliance_rate = 90  # Target for compliance (more than 90% compliance rate)

        # Custom color mappings for violation and compliance types
        violation_colors = {
            "No Helmet": "#d62728",  # Red
            "No Vest": "#9467bd",  # Purple
            "No Gloves": "#2ca02c",  # Green
            "No Goggles": "#ff7f0e",  # Orange
            "Other": "#1f77b4"  # Blue for any other violation types
        }

        compliance_colors = {
            "Compliant": "#2ca02c",  # Green for compliance
        }

        # Filter data and display insights based on the selected analysis type
        if analysis_type == "Violation":
            st.subheader("Violation Target is 10%")

            # Filter for Violations
            violation_data = data[data['Violation_Type'] != 'Compliant']

            # Create a Difference Chart for Violations
            if not violation_data.empty:
                violation_count = violation_data['Violation_Type'].value_counts().reset_index()
                violation_count.columns = ['Violation Type', 'Count']

                # Current Violation Rate Calculation
                current_violation_rate = (violation_count['Count'].sum() / data.shape[0]) * 100 if data.shape[
                                                                                                       0] > 0 else 0

                # Create a Difference Chart
                fig_difference = go.Figure()
                fig_difference.add_trace(go.Bar(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_violation_rate, target_violation_rate],
                    name='Rates',
                    marker_color=['#1f77b4', '#ff7f0e'],  # Blue for current rate, orange for target
                ))

                # Add Line for Difference
                fig_difference.add_trace(go.Scatter(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_violation_rate, target_violation_rate],
                    mode='lines+text',
                    name='Difference',
                    text=[f"{current_violation_rate:.2f}%", f"{target_violation_rate:.2f}%"],
                    textposition='top center',
                    line=dict(color='red', width=2)  # Red line for difference
                ))

                # Update layout for the difference chart
                fig_difference.update_layout(
                    title='Current vs. Target Violation Rate',
                    xaxis_title='Rate Type',
                    yaxis_title='Rate (%)',
                    showlegend=True,
                    plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
                )

                # Display Difference Chart
                st.plotly_chart(fig_difference, use_container_width=True)

                # Create a Bar Chart for Violation Counts with custom colors
                violation_count['Color'] = violation_count['Violation Type'].map(violation_colors)
                fig_violation = px.bar(violation_count, x='Violation Type', y='Count',
                                       title="Violation Counts",
                                       labels={'Count': 'Number of Violations'},
                                       color='Violation Type',
                                       color_discrete_map=violation_colors)  # Assign specific colors to each violation type

                # Update layout for violation chart
                fig_violation.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                    font=dict(color="white"),  # White font for contrast
                )

                # Display Violation Chart
                st.plotly_chart(fig_violation, use_container_width=True)

            else:
                st.write("No violation data available for the selected filters.")

        elif analysis_type == "Compliance":
            st.subheader("Compliance Target is 90%")

            # Filter for Compliance
            compliance_data = data[data['Violation_Type'] == 'Compliant']

            # Create a Difference Chart for Compliance
            if not compliance_data.empty:
                compliance_count = compliance_data['Violation_Type'].value_counts().reset_index()
                compliance_count.columns = ['Compliance Type', 'Count']

                # Current Compliance Rate Calculation
                current_compliance_rate = (compliance_count['Count'].sum() / data.shape[0]) * 100 if data.shape[
                                                                                                         0] > 0 else 0

                # Create a Difference Chart for Compliance
                fig_difference_compliance = go.Figure()
                fig_difference_compliance.add_trace(go.Bar(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_compliance_rate, target_compliance_rate],
                    name='Rates',
                    marker_color=['#2ca02c', '#ff7f0e'],  # Green for current rate, orange for target
                ))

                # Add Line for Difference
                fig_difference_compliance.add_trace(go.Scatter(
                    x=['Current Rate', 'Target Rate'],
                    y=[current_compliance_rate, target_compliance_rate],
                    mode='lines+text',
                    name='Difference',
                    text=[f"{current_compliance_rate:.2f}%", f"{target_compliance_rate:.2f}%"],
                    textposition='top center',
                    line=dict(color='red', width=2)  # Red line for difference
                ))

                # Update layout for compliance difference chart
                fig_difference_compliance.update_layout(
                    title='Current vs. Target Compliance Rate',
                    xaxis_title='Rate Type',
                    yaxis_title='Rate (%)',
                    showlegend=True,
                    plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
                )

                # Display Difference Chart
                st.plotly_chart(fig_difference_compliance, use_container_width=True)

                # Create a Bar Chart for Compliance Counts with custom colors
                compliance_count['Color'] = compliance_count['Compliance Type'].map(compliance_colors)
                fig_compliance = px.bar(compliance_count, x='Compliance Type', y='Count',
                                        title="Compliance Counts",
                                        labels={'Count': 'Number of Compliant Cases'},
                                        color='Compliance Type',
                                        color_discrete_map=compliance_colors)  # Assign specific colors to compliance

                # Update layout for compliance chart
                fig_compliance.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                    font=dict(color="white"),  # White font for contrast
                )

                # Display Compliance Chart
                st.plotly_chart(fig_compliance, use_container_width=True)

            else:
                st.write("No compliance data available for the selected filters.")


    # Time Tracking Insights
    elif selected_insight == "Time Tracking Insights":

        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

        # Group data by Date and calculate violation/compliance rates over time
        if analysis_type == "Violation":
            st.subheader("Time Tracking Insights - Violation Rate Over Time")

            # Calculate daily violation rates
            time_tracking_violations = filtered_data.groupby('Date')['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

            st.dataframe(time_tracking_violations)

            # Line chart to track violations over time
            fig_time_tracking_violations = px.line(time_tracking_violations,
                                                   x='Date',
                                                   y='Violation Rate',
                                                   title="Violation Rate Over Time",
                                                   labels={'Violation Rate': 'Violation Rate (%)'},
                                                   line_shape='spline',  # Smooth curve
                                                   markers=True,  # Show markers on data points
                                                   color_discrete_sequence=px.colors.qualitative.Dark24)

            fig_time_tracking_violations.update_layout(
                xaxis_title="Date",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100])  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_time_tracking_violations, use_container_width=True)

        else:
            st.subheader("Time Tracking Insights - Compliance Rate Over Time")

            # Calculate daily compliance rates
            time_tracking_compliance = filtered_data.groupby('Date')['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

            st.dataframe(time_tracking_compliance)

            # Line chart to track compliance over time
            fig_time_tracking_compliance = px.line(time_tracking_compliance,
                                                   x='Date',
                                                   y='Compliance Rate',
                                                   title="Compliance Rate Over Time",
                                                   labels={'Compliance Rate': 'Compliance Rate (%)'},
                                                   line_shape='spline',  # Smooth curve
                                                   markers=True,  # Show markers on data points
                                                   color_discrete_sequence=px.colors.qualitative.Dark24)

            fig_time_tracking_compliance.update_layout(
                xaxis_title="Date",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100])  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_time_tracking_compliance, use_container_width=True)


    # Shift Productivity Insights
    elif selected_insight == "Shift Productivity Insights":
        st.subheader("Shift Productivity Insights")
        # Shift Productivity Insights for Violations and Compliance

        # Group data by Shift, Factory, and Department to analyze productivity for each shift
        if analysis_type == "Violation":
            st.subheader("Shift Productivity Insights - Violation Rate by Shift")

            # Calculate violation rates for each shift
            shift_violations = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x != 'Compliant').sum() / len(x) * 100).reset_index(name='Violation Rate')

            st.dataframe(shift_violations)

            # Bar chart to visualize violation rates by shift
            fig_shift_violations = px.bar(shift_violations,
                                          x='Shift',
                                          y='Violation Rate',
                                          color='Factory',
                                          facet_col='Department',  # Separate charts for each department
                                          title="Violation Rate by Shift",
                                          labels={'Violation Rate': 'Violation Rate (%)'},
                                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig_shift_violations.update_layout(
                xaxis_title="Shift",
                yaxis_title="Violation Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_shift_violations, use_container_width=True)

        else:
            st.subheader("Shift Productivity Insights - Compliance Rate by Shift")

            # Calculate compliance rates for each shift
            shift_compliance = filtered_data.groupby(['Shift', 'Factory', 'Department'])['Violation_Type'].apply(
                lambda x: (x == 'Compliant').sum() / len(x) * 100).reset_index(name='Compliance Rate')

            st.dataframe(shift_compliance)

            # Bar chart to visualize compliance rates by shift
            fig_shift_compliance = px.bar(shift_compliance,
                                          x='Shift',
                                          y='Compliance Rate',
                                          color='Factory',
                                          facet_col='Department',  # Separate charts for each department
                                          title="Compliance Rate by Shift",
                                          labels={'Compliance Rate': 'Compliance Rate (%)'},
                                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig_shift_compliance.update_layout(
                xaxis_title="Shift",
                yaxis_title="Compliance Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0 to 100%
            )

            st.plotly_chart(fig_shift_compliance, use_container_width=True)


    # Predictive Insights (Example)
    elif selected_insight == "Predictive Insights":
        st.subheader(f"{analysis_type} Monthly Rates and Prediction")
        # Create Line Chart for Monthly Rates and Prediction
        col1, col2, col3, col4 = st.columns(4)
        total_checks = filtered_data.shape[0]
        # Display Current Rate, Next Month Prediction, Total Checks, and Relevant Checks
        col1.metric(rate_label, f"{current_rate:.2f}%")
        col2.metric("Next Month Prediction", f"{next_month_prediction:.2f}%")
        col3.metric("Total Checks", total_checks)
        col4.metric("Relevant Checks", relevant_checks)
        # Create Combined Chart for Monthly and Predicted Rate
        fig_combined = go.Figure()

        # Add Monthly Rates
        fig_combined.add_trace(go.Scatter(
            x=monthly_rate['Month'].astype(str),  # Ensuring x-axis is in string format
            y=monthly_rate['Rate'],
            mode='lines+markers',
            name='Monthly Rate',
            line=dict(color='royalblue', width=3),  # Thicker, smooth line for clarity
            marker=dict(size=10, color='lightblue', symbol='circle', line=dict(width=2, color='royalblue'))
            # Distinct markers
        ))

        # Prepare the next month label
        next_month_label = pd.to_datetime(monthly_rate['Month'].iloc[-1]).to_period(
            'M').to_timestamp() + pd.offsets.MonthEnd(1)

        # Add Predicted Rate
        fig_combined.add_trace(go.Scatter(
            x=[*monthly_rate['Month'].astype(str), next_month_label.strftime('%Y-%m')],
            y=[*monthly_rate['Rate'], next_month_prediction],
            mode='lines+markers+text',
            name='Predicted Rate',
            text=[*[''] * len(monthly_rate), f"{next_month_prediction:.2f}%"],
            textposition='top center',
            line=dict(color='orange', dash='dash', width=3),  # Dashed line for predicted trend
            marker=dict(size=10, color='orange', symbol='diamond', line=dict(width=2, color='darkorange'))
            # Diamond markers
        ))

        # Add a shaded region for the prediction
        fig_combined.add_vrect(
            x0=monthly_rate['Month'].iloc[-1], x1=next_month_label.strftime('%Y-%m'),
            fillcolor="lightgray", opacity=0.3, line_width=0,
            annotation_text="Prediction Area", annotation_position="top left"
        )

        # Update layout for the combined chart
        fig_combined.update_layout(
            title=f"{analysis_type} Rate Over Time with Forecast",
            xaxis_title='Month',
            yaxis_title='Rate (%)',
            xaxis=dict(tickvals=[*monthly_rate['Month'].astype(str), next_month_label.strftime('%Y-%m')]),
            showlegend=True,
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            paper_bgcolor='rgba(255, 255, 255, 0)',  # Keep paper background white
            font=dict(color="black"),
        )

        # Display Combined Chart
        st.plotly_chart(fig_combined, use_container_width=True)

        # Create Difference Chart
        if monthly_rate.shape[0] > 0:
            difference = next_month_prediction - current_rate

            fig_difference = go.Figure()

            # Add Current vs Predicted Rate Bars
            fig_difference.add_trace(go.Bar(
                x=['Current Rate', 'Next Month Prediction'],
                y=[current_rate, next_month_prediction],
                name='Rates',
                marker_color=['royalblue', 'orange'],  # Color difference between current and predicted
                text=[f"{current_rate:.2f}%", f"{next_month_prediction:.2f}%"],
                textposition='auto',  # Auto place percentage values on bars
            ))

            # Add a connecting line for Difference
            fig_difference.add_trace(go.Scatter(
                x=['Current Rate', 'Next Month Prediction'],
                y=[current_rate, next_month_prediction],
                mode='lines+text',
                name='Difference',
                text=[f"{difference:.2f}%" if difference > 0 else f"{-difference:.2f}% down", ''],
                textposition='top center',
                line=dict(color='red', width=3, dash='dot')  # Use red dotted line for difference
            ))

            # Update layout for the difference chart
            fig_difference.update_layout(
                title='Current vs. Predicted Rate Difference',
                xaxis_title='Rate Type',
                yaxis_title='Rate (%)',
                showlegend=False,  # No need for legend here
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font=dict(color="black"),
            )

            # Display Difference Chart
            st.plotly_chart(fig_difference, use_container_width=True)

        # Table: Display Current and Predicted Rates
        st.table(pd.DataFrame({
            'Rate Type': ['Current Rate', 'Next Month Prediction'],
            'Value (%)': [current_rate, next_month_prediction]
        }))        # Combined Chart

    # Growth Tracker Insights
    elif selected_insight == "Growth Tracker Insights":

        st.subheader("Growth Tracker Insights")


        # Calculate the overall compliance checks from the entire dataset
        overall_compliant_checks = filtered_data[filtered_data['Violation_Type'] == 'Compliant'].shape[0]
        overall_total_checks = filtered_data.shape[0]

        # Calculate the current overall compliance rate
        current_overall_rate = (
                    overall_compliant_checks / overall_total_checks * 100) if overall_total_checks > 0 else 0

        # Display the growth tracker insights
        st.subheader(f"Growth Tracker Insights - {analysis_type} Improvement")

        overall_total_checks = filtered_data.shape[0]

        # Calculate the current overall compliance rate
        current_overall_rate = (
                    overall_compliant_checks / overall_total_checks * 100) if overall_total_checks > 0 else 0



        # Create a DataFrame for growth tracking
        monthly_rate = pd.DataFrame({'Month': ['Current'], 'Current Rate': [current_rate]})

        # For this example, we're using only the current rate; you can adjust this as needed
        monthly_rate['Growth Rate'] = monthly_rate['Current Rate'].diff().fillna(
            0)  # Growth from previous month (if applicable)

        # Display the monthly rates and growth rates for reference
        st.dataframe(monthly_rate)
        # Calculate the overall compliance checks from the entire dataset
        overall_compliant_checks = filtered_data[filtered_data['Violation_Type'] == 'Compliant'].shape[0]
        overall_total_checks = filtered_data.shape[0]

        # Calculate the current overall compliance rate
        current_overall_rate = (
                overall_compliant_checks / overall_total_checks * 100) if overall_total_checks > 0 else 0


        # Create a DataFrame for growth tracking with the current overall rate
        if analysis_type == "Violation":
            current_rate = (filtered_data[filtered_data['Violation_Type'] != 'Compliant'].shape[
                                0] / overall_total_checks * 100) if overall_total_checks > 0 else 0
        else:
            current_rate = current_overall_rate

        # Ensure relevant data is present for growth tracking
        if overall_total_checks > 0:
            # Create a DataFrame for growth tracking with current and previous rates
            current_month_data = pd.DataFrame({'Month': ['Current'], 'Current Rate': [current_rate]})

            # For previous month rate, use a placeholder or calculate it if available
            previous_month_rate = 5.0  # Replace with actual previous month's data if available
            previous_month_data = pd.DataFrame({'Month': ['Previous'], 'Current Rate': [previous_month_rate]})

            # Concatenate the current and previous month DataFrames
            monthly_rate = pd.concat([previous_month_data, current_month_data], ignore_index=True)

            # Calculate the growth rate
            monthly_rate['Growth Rate'] = monthly_rate['Current Rate'].diff().fillna(0)  # Growth from previous month

            # Calculate the overall improvement trend
            total_growth = monthly_rate['Growth Rate'].sum()
            trend_message = f"Overall {analysis_type} Growth: {total_growth:.2f}% from Previous to Current"

            # Plot the growth rate over time
            fig_growth = px.line(monthly_rate, x='Month', y='Current Rate',
                                 title=f"Current {analysis_type} Rate Over Time",
                                 labels={'Current Rate': f'{analysis_type} Rate (%)'},
                                 color_discrete_sequence=['green' if total_growth > 0 else 'red'])

            # Add scatter points to highlight individual growth points
            fig_growth.add_scatter(x=monthly_rate['Month'], y=monthly_rate['Current Rate'],
                                   mode='markers', name='Rate Points',
                                   marker=dict(color='blue', size=8, symbol='circle'))

            # Update layout for better display
            fig_growth.update_layout(
                xaxis_title="Month",
                yaxis_title=f"{analysis_type} Rate (%)",
                yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0% to 100%
            )

            # Display the growth rate chart
            st.plotly_chart(fig_growth, use_container_width=True)

        else:
            st.write(f"No data available to calculate {analysis_type} Growth Rate.")


    # Risk Radar Insights
    elif selected_insight == "Risk Radar Insights":
        # Risk Radar Insights Section
        st.subheader("Risk Radar Insights")

        # Calculate overall compliance and violation rates
        overall_compliance_rate = (
                    filtered_data[filtered_data['Violation_Type'] == 'Compliant'].shape[0] / filtered_data.shape[
                0] * 100) if filtered_data.shape[0] > 0 else 0
        overall_violation_rate = (
                    filtered_data[filtered_data['Violation_Type'] != 'Compliant'].shape[0] / filtered_data.shape[
                0] * 100) if filtered_data.shape[0] > 0 else 0

        # Create a DataFrame for visualization
        risk_data = pd.DataFrame({
            'Category': ['Compliance', 'Violation'],
            'Rate': [overall_compliance_rate, overall_violation_rate]
        })

        # Create a line chart for risk insights
        fig_risk = px.line(risk_data, x='Category', y='Rate', title='Risk Radar Insights',
                           labels={'Rate': 'Rate (%)'},
                           color_discrete_sequence=['green' if overall_compliance_rate >= 80 else 'red',
                                                    'red' if overall_violation_rate >= 80 else 'green'])

        # Add annotations for color meanings
        fig_risk.add_annotation(
            x=0, y=overall_compliance_rate + 5,
            text="Normal (Green: ≥ 80%)",
            showarrow=False,
            font=dict(color='green', size=12)
        )
        fig_risk.add_annotation(
            x=1, y=overall_violation_rate + 5,
            text="Risk (Red: > 80%)",
            showarrow=False,
            font=dict(color='red', size=12)
        )

        # Update layout for better display
        fig_risk.update_layout(
            yaxis=dict(range=[0, 100]),  # Set Y-axis range from 0% to 100%
            xaxis_title="Category",
            yaxis_title="Rate (%)"
        )

        # Display the risk radar chart
        st.plotly_chart(fig_risk, use_container_width=True)

        # Check for departments with low performance
        low_performance_departments = filtered_data.groupby('Department')['Violation_Type'].value_counts(
            normalize=True).unstack().fillna(0)
        low_performance_departments['Compliance Rate'] = low_performance_departments.get('Compliant', 0) * 100
        low_performance_departments['Violation Rate'] = low_performance_departments.get('Compliant', 0).apply(
            lambda x: 100 - x)

        # Filter departments based on low performance thresholds
        low_performance_departments = low_performance_departments[
            (low_performance_departments['Compliance Rate'] < 80) | (
                        low_performance_departments['Violation Rate'] > 80)]

        # Display low performance departments
        if not low_performance_departments.empty:
            st.write("Departments with Low Performance:")
            st.dataframe(low_performance_departments[['Compliance Rate', 'Violation Rate']])
        else:
            st.write("All departments are performing well.")





    # Association Insights
    elif selected_insight == "Association Insights":
        st.subheader("Association Insights")


        # Filter data based on the selected analysis type (Violation or Compliance)
        if analysis_type == "Violation":
            relevant_data = filtered_data[filtered_data['Violation_Type'] != 'Compliant']
        else:
            relevant_data = filtered_data[filtered_data['Violation_Type'] == 'Compliant']

        # Count violations by department
        violation_counts = relevant_data.groupby(['Department', 'Violation_Type']).size().reset_index(name='Counts')

        # Create a bar chart
        fig_violation_department = px.bar(violation_counts,
                                          x='Department',
                                          y='Counts',
                                          color='Violation_Type',
                                          title="Violations by Department",
                                          labels={'Counts': 'Number of Violations'},
                                          color_discrete_sequence=px.colors.qualitative.Set1)

        # Update layout
        fig_violation_department.update_layout(xaxis_title="Department",
                                               yaxis_title="Number of Violations",
                                               yaxis=dict(range=[0, violation_counts['Counts'].max() + 5]))

        # Display the chart
        st.plotly_chart(fig_violation_department, use_container_width=True)
        # Create a pivot table for the heatmap
        heatmap_data = relevant_data.pivot_table(index='Department', columns='Shift',
                                                 values='Violation_Type', aggfunc='count', fill_value=0)

        # Create a heatmap with a different color scale
        fig_heatmap = px.imshow(
            heatmap_data,
            title="Heatmap of Violations by Shift and Department",
            labels=dict(x="Shift", y="Department", color="Number of Violations"),
            color_continuous_scale='Blues',  # Different color scale: Blues
            aspect="auto"  # Adjust aspect ratio for better visibility
        )

        # Update layout for better aesthetics
        fig_heatmap.update_layout(
            xaxis_title="Shift",
            yaxis_title="Department",
            coloraxis_colorbar=dict(title="Number of Violations"),
            plot_bgcolor='rgba(0,0,0,0)'  # Optional: make background transparent
        )

        # Display the heatmap
        st.plotly_chart(fig_heatmap, use_container_width=True)
        # Count the occurrences of each violation type
        violation_type_counts = relevant_data['Violation_Type'].value_counts().reset_index()
        violation_type_counts.columns = ['Violation_Type', 'Counts']

        # Create a pie chart
        fig_violation_pie = px.pie(violation_type_counts,
                                   names='Violation_Type',
                                   values='Counts',
                                   title="Distribution of Violation Types",
                                   color_discrete_sequence=px.colors.sequential.RdBu)

        # Display the pie chart
        st.plotly_chart(fig_violation_pie, use_container_width=True)

