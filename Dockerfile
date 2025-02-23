# get python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src and data folder into the container
COPY src ./src
COPY data ./data

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
