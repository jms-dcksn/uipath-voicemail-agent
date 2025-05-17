from uipath import UiPath
import os

def download_file(blob_file_path: str, name: str, folder_path: str, destination_path: str = None) -> str:
    """
    Download a file from UiPath bucket to a specified or temporary location.
    
    Args:
        blob_file_path: Path to the file in the bucket (typically the filename)
        name: Name of the bucket
        folder_path: OR folder path where bucket is located
        destination_path: Optional path where to save the file. If None, saves to /tmp
    
    Returns:
        str: The path where the file was saved
    """
    # Handle the case where no destination path is provided
    if destination_path is None:
        filename = os.path.basename(blob_file_path)
        destination_path = os.path.join('/tmp', filename)
    
    # Initialize client and perform download
    client = UiPath()
    client.buckets.download(
        name=name,
        blob_file_path=blob_file_path,
        destination_path=destination_path,
        folder_path=folder_path
    )
    return destination_path

# Example usage:
# download_file(
#     blob_file_path="TechnicalRequirements.csv",
#     destination_path="/Users/jamesdickson/Desktop/downloaded_data.csv",
#     name="Technical Requirements",
#     folder_path="Demos/EFX"
# )

