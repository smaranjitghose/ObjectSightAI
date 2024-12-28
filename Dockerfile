# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# healthcheck to verify the service is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]