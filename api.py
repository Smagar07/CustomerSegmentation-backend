import requests

url = 'http://127.0.0.1:8000/cluster'  # Update with your API URL

# Prepare the CSV file
files = {'file': open('Customer_Data.csv', 'rb')}  # Provide the path to your CSV file

# Send the request
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    # Save the returned clustered image
    with open('clustered_image.png', 'wb') as f:
        f.write(response.content)
    print("Clustered image saved successfully.")
else:
    print("Error:", response.text)
