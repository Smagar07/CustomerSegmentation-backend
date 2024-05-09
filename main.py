from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import io
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Load the KMeans model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Drop unnecessary columns
    df = df.drop(columns=["CustomerID", "Gender"])

    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Define cluster numbers
    clusters = [2, 3, 4, 5]
    
    # Create a single figure containing subplots for each cluster number
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for i, num_clusters in enumerate(clusters):
        # Fit the model for current number of clusters
        kmeans_model.set_params(n_clusters=num_clusters)
        cluster_labels = kmeans_model.fit_predict(df_scaled)
        df['cluster'] = cluster_labels

        
        # Generate visualization
        palette = sns.color_palette("hls", num_clusters)
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', data=df, palette=palette, s=100, ax=axs[i//2, i%2])
        axs[i//2, i%2].set_title(f'k: {num_clusters}')
        axs[i//2, i%2].set_xlabel('Annual Income (k$)')
        axs[i//2, i%2].set_ylabel('Spending Score (1-100)')
        axs[i//2, i%2].legend(title='Cluster')
        axs[i//2, i%2].grid(True)

    # Save the figure to a bytes object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Clear the figure to avoid overlapping when multiple requests are made
    plt.clf()
    plt.close(fig)
    
    # Return the plot as a streaming response
    return StreamingResponse(content=buf, media_type="image/png")
