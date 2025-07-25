# LiveInsight: Real-Time Retail Analytics Dashboard

LiveInsight is a comprehensive, real-time data analytics platform designed to monitor and visualize retail performance and system health. It leverages a streaming data pipeline built with Python, Apache Spark, and Streamlit to provide instantaneous insights into sales, inventory, and system resource utilization.

## Project Overview

This project simulates a live stream of retail sales data, processes it in real-time using Spark Streaming to calculate key business metrics, and displays these insights on an interactive web-based dashboard. A key feature is the integration of system hardware monitoring (CPU/GPU temperature and usage) and a machine-learning-driven predictive inventory management system.

### Core Components

1. **Data Stream Server (`stream_server.py`)**: A Python socket server that reads retail data from a CSV file and streams it line-by-line, simulating a continuous flow of sales transactions.
2. **Spark Streaming Processor (`spark_streaming.py`)**: An Apache Spark application that consumes the data stream, performs stateful aggregations in real-time, and saves the processed results into structured CSV files.
3. **Streamlit Dashboard (`dashboard.py`)**: A rich, interactive web dashboard that visualizes the processed data, monitors system health, and provides predictive alerts for inventory management.

## Features

* **Real-Time KPI Tracking**: Monitor total revenue, units sold, and transaction counts as they happen.
* **Performance Dashboards**: Interactive charts for sales by branch, product category, top-selling products, and payment methods.
* **Time-Series Analysis**: Track weekly and monthly revenue trends for each branch.
* **System Health Monitoring**: Live gauges and charts for CPU/GPU temperature, CPU/Memory/Disk usage.
* **Predictive Inventory Management**: An ML model predicts stock depletion rates based on recent sales velocity, issuing alerts for items that need reordering.
* **Business Alerts**: Automated notifications for low-selling products and underperforming branches.
* **Demo Mode**: A feature to generate sample data for demonstration purposes when a live data source isn't available.

## Project Architecture

1. `stream_server.py` reads data from `retail_data_bangalore.csv`.
2. The server sends this data over a TCP socket (`localhost:9999`).
3. `spark_streaming.py` connects to the socket, ingests the data stream, and performs aggregations.
4. The aggregated data (e.g., total sales per branch) is continuously saved to the `output/` directory.
5. `dashboard.py` reads the aggregated data from the `output/` directory, displays it, and auto-refreshes to show the latest insights.

## Getting Started

### Prerequisites

* **Python 3.8+**
* **Java 8 or 11** (for Apache Spark)
* **Apache Spark 3.x**
* **A sample CSV dataset** named `retail_data_bangalore.csv` in the root directory. You can use any sales data, but ensure the columns match the expected format.

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. **Install Python dependencies:**
   It is highly recommended to use a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content:

   ```
   pandas
   pyspark
   streamlit
   psutil
   plotly
   scikit-learn
   # For Windows temperature monitoring (optional)
   WMI
   pywin32
   ```

3. **Install Apache Spark**:

   * Download Spark from the official website
   * Extract the archive and set the `SPARK_HOME` environment variable to the extracted directory path.
   * Add Spark's `bin` directory to your system's `PATH`.

### Running the Project

The application must be launched in a specific order. Open three separate terminal windows.

**Terminal 1: Start the Spark Streaming Processor**

```bash
spark-submit spark_streaming.py
```

You should see log messages indicating that the Spark context has started and is waiting for data.

**Terminal 2: Start the Data Stream Server**

```bash
python stream_server.py
```

You will see the server start listening and then begin sending data records from the CSV file.

**Terminal 3: Launch the Streamlit Dashboard**

```bash
streamlit run dashboard.py
```

Your default web browser should open with the LiveInsight dashboard. The charts will start populating as the data is processed by Spark.

## File Descriptions

### stream\_server.py

* Uses socket to create a TCP server.
* Reads the `retail_data_bangalore.csv` file using pandas.
* Cleans and formats data on-the-fly (e.g., timestamps, integer conversion).
* Sends each row as a comma-separated string to any connected client (the Spark script).

### spark\_streaming.py

* Initializes a SparkContext and StreamingContext.
* Connects to the socket stream from `stream_server.py`.
* Parses each incoming line into a structured Row object, handling potential errors gracefully.
* Uses `updateStateByKey` for stateful aggregations, allowing it to maintain running totals for revenue, sales counts, etc.
* Defines multiple aggregation streams for branch sales, category sales, product sales, payment types, and time-based revenue.
* Saves the output of each stream into its respective folder within the `output/` directory.

### dashboard.py

* Built with Streamlit for the user interface.
* Loads the aggregated data from the `output/` directory using pandas.
* Uses Plotly to create interactive and visually appealing charts.
* The dashboard is set to auto-refresh every few seconds to provide a live view of the data.
* Integrates psutil and WMI (on Windows) to fetch and display real-time system resource usage and temperatures.
* Contains a predictive inventory section that:

  * Calculates the current stock based on simulated starting inventory and sales.
  * Trains a RandomForestRegressor model to predict how many days are left before a product runs out of stock.
  * Displays inventory levels with progress bars and provides urgent reorder alerts.

## Customization

* **Data Source**: To use your own data, replace `retail_data_bangalore.csv` and modify the parsing logic in `spark_streaming.py` and `stream_server.py` to match your schema.
* **Refresh Interval**: Change the auto-refresh speed from the sidebar control on the dashboard.
* **Alert Thresholds**: Adjust the logic for low-selling products or underperforming branches directly in `dashboard.py`.
* **ML Model**: The inventory prediction model in `dashboard.py` can be replaced with a more sophisticated one or trained on more extensive historical data.
